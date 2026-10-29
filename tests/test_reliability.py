"""Runtime regressions: decoding, bounded OCR, uploads, locks and crop safeguards."""
import io
import shutil
import subprocess
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import server
from yunost.layout import LayoutBlock, LayoutWord
from yunost.ocr import OCRTimeout, TesseractEngine, ocr_budget
from yunost.pipeline import UnreadableImage, load_page
from yunost.refinement import refine_words
from yunost.text import TextPageResult


def encoded(image, fmt='PNG', **kwargs):
    out = io.BytesIO()
    image.save(out, format=fmt, **kwargs)
    return out.getvalue()


@pytest.mark.parametrize('mode', ['RGBA', 'LA', 'P'])
def test_transparent_background_is_white_not_black(mode):
    image = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
    if mode == 'LA':
        image = image.convert('LA')
    elif mode == 'P':
        image = image.convert('P')
    assert load_page(encoded(image)).min() == 255


def test_alpha_text_survives_on_white_paper():
    image = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
    image.paste((0, 0, 0, 255), (20, 20, 80, 80))
    page = load_page(encoded(image))
    assert page[1000, 1000] == 0 and page[50, 50] == 255


def test_exif_rotation_applies_before_normalized_geometry():
    image = Image.new('RGB', (200, 100), 'white')
    exif = image.getexif()
    exif[274] = 6
    assert load_page(encoded(image, 'JPEG', exif=exif)).shape == (2000, 1000)


def test_wide_page_has_bounded_normalized_memory():
    page = load_page(encoded(Image.new('L', (600, 100), 255)))
    assert page.shape == (667, 4000)
    assert page.size <= 8_000_000


def test_animation_is_not_silently_truncated_to_first_frame():
    frames = [Image.new('RGB', (100, 100), color) for color in ['white', 'black']]
    with pytest.raises(UnreadableImage, match='одну страницу'):
        load_page(encoded(frames[0], 'GIF', save_all=True, append_images=frames[1:]))


def test_truncated_image_returns_readable_error():
    data = encoded(Image.new('RGB', (200, 200), 'white'))
    with pytest.raises(UnreadableImage):
        load_page(data[:len(data) // 2])


def test_over_pixel_limit_is_rejected_before_decoding(monkeypatch):
    monkeypatch.setattr(Image, 'MAX_IMAGE_PIXELS', 10_000)
    with pytest.warns(Image.DecompressionBombWarning):
        with pytest.raises(UnreadableImage):
            load_page(encoded(Image.new('L', (110, 100))))


def empty_page():
    return TextPageResult([], np.full((200, 200), 255, np.uint8))


def test_api_rejects_oversize_body_before_ocr(monkeypatch):
    monkeypatch.setattr(server, 'MAX_UPLOAD', 8)
    response = TestClient(server.app).post('/api/ocr', files={'file': ('a.png', b'a' * 9)})
    assert response.status_code == 413


def test_api_rejects_chunked_multipart_and_closes_spool(monkeypatch):
    monkeypatch.setattr(server, 'MAX_UPLOAD', 8)
    boundary = 'regression-boundary'
    def chunks():
        yield (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.png"\r\n\r\n').encode()
        for _ in range(80):
            yield b'a' * 1024
        yield f'\r\n--{boundary}--\r\n'.encode()
    response = TestClient(server.app).post('/api/ocr', content=chunks(),
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    assert response.status_code == 413


def test_body_content_length_early_limit(monkeypatch):
    monkeypatch.setattr(server, 'MAX_UPLOAD', 8)
    response = TestClient(server.app).post('/api/ocr', content=b'a' * 70_000)
    assert response.status_code == 413


def test_bad_content_length_is_rejected():
    response = TestClient(server.app).post('/api/ocr', content=b'', headers={'Content-Length': '-1'})
    assert response.status_code == 400


def test_health_reports_tesseract_failure(monkeypatch):
    monkeypatch.setattr(server.shutil, 'which', lambda _: '/fake/tesseract')
    def timeout(*a, **kw):
        raise subprocess.TimeoutExpired('tesseract', 5)
    monkeypatch.setattr(server.subprocess, 'run', timeout)
    assert server.health()['status'] == 'tesseract_error'


def test_busy_server_does_not_start_second_ocr():
    assert server._ocr_slot.acquire(blocking=False)
    try:
        with TestClient(server.app) as client:
            response = client.post('/api/ocr', files={'file': ('a.png', b'fixture')})
            assert response.status_code == 429
            assert response.headers['retry-after'] == '3'
            assert client.get('/api/health').status_code == 200
    finally:
        server._ocr_slot.release()


@pytest.mark.parametrize('exception,status', [(OCRTimeout('timeout'), 504),
    (UnreadableImage('bad image'), 400), (RuntimeError('broken OCR'), 500)])
def test_ocr_failure_releases_slot_and_next_request_works(monkeypatch, exception, status):
    def broken(_):
        raise exception
    monkeypatch.setattr(server, 'get_extractor', lambda: SimpleNamespace(extract=broken))
    with TestClient(server.app) as client:
        assert client.post('/api/ocr', files={'file': ('a.png', b'fixture')}).status_code == status
        monkeypatch.setattr(server, 'get_extractor', lambda: SimpleNamespace(extract=lambda _: empty_page()))
        response = client.post('/api/ocr', files={'file': ('a.png', b'fixture')})
        assert response.status_code == 200
        assert response.json()['text_blocks'] == 0


def test_tesseract_process_timeout_is_explicit(monkeypatch):
    engine = TesseractEngine(timeout=0.01)
    def timeout(*args, **kwargs):
        assert 0 < kwargs['timeout'] <= 0.01
        raise RuntimeError('Tesseract process timeout')
    monkeypatch.setattr(engine._tess, 'image_to_data', timeout)
    with pytest.raises(OCRTimeout):
        engine.read_layout(np.zeros((64, 64), np.uint8))


@pytest.mark.skipif(shutil.which('tesseract') is None, reason='Tesseract not installed')
def test_real_tesseract_subprocess_is_killed_on_timeout():
    engine = TesseractEngine(timeout=0.000001)
    with pytest.raises(OCRTimeout):
        engine.read_layout(np.full((2000, 1500), 255, np.uint8))


def test_whole_page_budget_expires_and_resets(monkeypatch):
    engine = TesseractEngine()
    monkeypatch.setattr(engine._tess, 'image_to_data', lambda *a, **kw: {'text': []})
    with ocr_budget(-1):
        with pytest.raises(OCRTimeout):
            engine.read_layout(np.zeros((64, 64), np.uint8))
    assert engine.read_layout(np.zeros((64, 64), np.uint8)) == []


def refinement_pair():
    words = [LayoutWord(text, 10 + i * 100, 10, 80, 20, 70, (1, 1, 1))
             for i, text in enumerate(['Тде', 'сохраняют', 'цвета', 'чистоту'])]
    block = LayoutBlock(words)
    reread = LayoutBlock([replace(w, text='Где' if i == 0 else w.text, confidence=95)
                           for i, w in enumerate(words)])
    return block, reread


def test_refinement_changes_only_spelling_preserves_geometry_and_original():
    block, reread = refinement_pair()
    box, ids = block.box, [w.line_id for w in block.words]
    assert refine_words(block, [reread], lambda *a: True) == 1
    assert block.words[0].text == 'Где' and block.words[0].original_text == 'Тде'
    assert block.box == box and [w.line_id for w in block.words] == ids
    assert block.to_dict()['refined_words'] == 1


@pytest.mark.parametrize('damage', ['missing_word', 'extra_word', 'different_lines',
                                   'moved_word', 'wrong_columns', 'low_confidence', 'rewrite'])
def test_refinement_refuses_layout_changes_and_uncertain_rewrites(damage):
    block, reread = refinement_pair()
    before = block.text
    if damage == 'missing_word':
        reread.words.pop()
    elif damage == 'extra_word':
        reread.words.append(replace(reread.words[-1]))
    elif damage == 'different_lines':
        reread.words[0].line_id = (1, 1, 2)
    elif damage in ['moved_word', 'wrong_columns']:
        reread.words[0].x += 100
    elif damage == 'low_confidence':
        for w in reread.words:
            w.confidence = 60
    elif damage == 'rewrite':
        for w in reread.words:
            w.text = 'другое'
    assert refine_words(block, [reread], lambda *a: True) == 0
    assert block.text == before


def test_refinement_needs_independent_word_confirmation():
    block, reread = refinement_pair()
    assert refine_words(block, [reread], lambda *a: False) == 0
    assert block.words[0].text == 'Тде'


def test_refinement_never_silently_changes_punctuation():
    block, reread = refinement_pair()
    block.words[0].text = 'Где,'
    reread.words[0].text = 'Где.'
    assert refine_words(block, [reread], lambda *a: True) == 0
    assert block.words[0].text == 'Где,'


def test_static_script_is_served_and_html_uses_it():
    with TestClient(server.app) as client:
        assert client.get('/health').json()['status'] == client.get('/api/health').json()['status']
        assert '/static/app.js' in client.get('/').text
        response = client.get('/static/app.js')
        assert response.status_code == 200 and '/api/ocr' in response.text
