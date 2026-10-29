"""Additive physical-edge diagnostics must never change the all-text OCR API."""
import base64
import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
from yunost.layout import LayoutBlock, LayoutWord
from yunost.text import TextPageResult


def block(text, y=10, rejected=''):
    return LayoutBlock([LayoutWord(line, 10, y + i * 25, 140, 15, 95, (1, 1, i))
                        for i, line in enumerate(text.splitlines())], rejected=rejected)


def page(blocks):
    return TextPageResult(blocks, np.full((300, 240), 255, np.uint8))


def old_fields(result):
    """Freeze the pre-edge API contract, including overlay bytes and word data."""
    return json.loads(json.dumps({
        'text': result.text,
        'text_blocks': len(result),
        'other_blocks': len(result.blocks) - len(result),
        'blocks': [b.to_dict() for b in result],
        'first_block': result[0].to_dict() if len(result) else None,
        'all_blocks': [b.to_dict() for b in result.blocks],
        'engine': 'tesseract',
        'mode': 'all_text',
        'page_size': {'width': result.gray.shape[1], 'height': result.gray.shape[0]},
        'overlay': 'data:image/png;base64,' + base64.b64encode(result.overlay_png()).decode('ascii'),
    }))


def analysis(blocks=(), value=None, warnings=()):
    return {'method': 'components-v2', 'normalization': 'padded_ocr_block_width',
            'blocks': list(blocks), 'value': value, 'warnings': list(warnings)}


def post_page(monkeypatch, result):
    # The bytes are deliberately not an image: neither OCR nor Tesseract is used.
    monkeypatch.setattr(server, 'get_extractor', lambda: SimpleNamespace(extract=lambda _: result))
    with TestClient(server.app) as client:
        response = client.post('/api/ocr', files={'file': ('page.png', b'fixture')})
    assert response.status_code == 200
    return response.json()


def test_edge_field_is_the_only_addition_and_uses_all_block_numbers(monkeypatch):
    result = page([block('Проза. Продолжение.'), block('?!', 45, rejected='шум'),
                   block('Стихи\nКороткая строка', 80), block('Я', 150)])
    before = old_fields(result)
    measured = analysis([
        {'block_number': 3, 'value': .125, 'line_count': 4, 'correction_angle': 1.5,
         'warnings': [], 'lines': [{'box': [10, 80, 150, 95], 'right': 149.5,
                                    'center_y': 87.5, 'aligned_right': 140}]},
        {'block_number': 1, 'value': None, 'line_count': 1, 'lines': [],
         'correction_angle': 0, 'warnings': ['fewer_than_four_lines']},
    ], value=.125)
    calls = []

    def measure(actual):
        calls.append(actual)
        return measured

    monkeypatch.setattr(server, 'analyze_page_edges', measure)
    body = post_page(monkeypatch, result)
    assert body.pop('edge_analysis') == measured
    assert body == before
    assert len(calls) == 1 and calls[0] is result
    assert old_fields(result) == before


@pytest.mark.parametrize('count,value', [(0, None), (1, None), (4, 0.0)])
def test_empty_null_and_measured_zero_are_distinct(monkeypatch, count, value):
    result = page([block('Я')]) if count else page([])
    measured = analysis([{'block_number': 1, 'value': value, 'line_count': count,
                          'lines': [], 'correction_angle': 0, 'warnings': []}]
                        if count else [], value=value)
    monkeypatch.setattr(server, 'analyze_page_edges', lambda _: measured)
    body = post_page(monkeypatch, result)
    returned = body.pop('edge_analysis')
    assert returned == measured
    assert body == old_fields(result)
    if value is None:
        assert returned['value'] is None
    else:
        assert returned['value'] == 0.0


def test_real_helper_handles_empty_page_without_ocr():
    body = server.page_response(page([]))
    assert body['edge_analysis'] == analysis()
    assert body['text'] == '' and body['blocks'] == [] and body['first_block'] is None


@pytest.mark.parametrize('text', ['Рассказ. Обычная проза.', 'Стих\nЕщё строка', 'Я', '2026', 'TITLE'])
@pytest.mark.parametrize('value', [None, 0.0, .7])
def test_edge_value_does_not_filter_any_text_or_assign_genre(monkeypatch, text, value):
    result = page([block(text)])
    measured = analysis([{'block_number': 1, 'value': value, 'line_count': 1,
                          'lines': [], 'correction_angle': 0, 'warnings': []}], value=value)
    monkeypatch.setattr(server, 'analyze_page_edges', lambda _: measured)
    body = post_page(monkeypatch, result)
    assert body.pop('edge_analysis') == measured
    assert body == old_fields(result)
    assert body['text'] == text and body['text_blocks'] == 1
    assert body['blocks'][0]['is_text']
    assert 'genre' not in body and 'is_prose' not in body['blocks'][0]


def test_geometry_failure_preserves_successful_ocr(monkeypatch):
    result = page([block('Проза'), block('Стихи\nЯ', 80)])
    before = old_fields(result)

    def broken(_):
        raise RuntimeError('geometry unavailable')

    monkeypatch.setattr(server, 'analyze_page_edges', broken)
    body = post_page(monkeypatch, result)
    assert body.pop('edge_analysis') == analysis(warnings=['edge_analysis_unavailable'])
    assert body == before


def test_local_ui_has_edge_toggle():
    with TestClient(server.app) as client:
        html = client.get('/').text
        script = client.get('/static/app.js').text
    assert 'id="show-edges" disabled' in html and '<canvas id="edge-canvas" hidden' in html
    assert 'edge_analysis' in script and 'page.page_size.width' in script


def test_page_helper_uses_accepted_blocks_and_within_block_rms(monkeypatch):
    import yunost.edges as edges
    result = page([block('Проза'), block('?!', 45, rejected='шум'),
                   block('Короткая строка', 80), block('Я', 150)])
    before = old_fields(result)
    calls = []

    def measure(mask, span, top, bottom):
        calls.append((span, top, bottom))
        count, value = [(4, .1), (12, .3), (1, None)][len(calls) - 1]
        return {'value': value, 'line_count': count, 'lines': []}

    monkeypatch.setattr(edges, 'measure_edges', measure)
    response = edges.analyze_page_edges(result)
    assert [b['block_number'] for b in response['blocks']] == [1, 3, 4]
    assert response['value'] == pytest.approx(((4 * .1**2 + 12 * .3**2) / 16)**.5)
    assert response['normalization'] == 'padded_ocr_block_width'
    assert calls == [((6, 154), 6, 29), ((6, 154), 76, 99), ((6, 154), 146, 169)]
    assert old_fields(result) == before
