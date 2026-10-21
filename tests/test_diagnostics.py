import io
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import server
from yunost.blocks import TextBlock
from yunost.pipeline import PageResult, ProseExtractor, UnreadableImage, load_page
from yunost.layout import LayoutBlock, LayoutWord
from yunost.text import TextPageResult


@pytest.mark.parametrize('languages,status', [
    ('List of available languages (1):\nrus\n', 'ok'),
    ('List of available languages (2):\neng\nosd\n', 'russian_missing')])
def test_health_checks_russian_weights(monkeypatch, languages, status):
    monkeypatch.setattr(server.shutil, 'which', lambda _: '/fake/tesseract')
    monkeypatch.setattr(server.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=languages))
    assert server.health()['status'] == status


def test_health_without_tesseract(monkeypatch):
    monkeypatch.setattr(server.shutil, 'which', lambda _: None)
    assert server.health()['status'] == 'tesseract_missing'


@pytest.mark.parametrize('size', [(10, 200), (2000, 100), (100, 2000)])
def test_reject_tiny_or_extreme_aspect_images(size):
    buffer = io.BytesIO()
    Image.new('L', size, 255).save(buffer, format='PNG')
    with pytest.raises(UnreadableImage):
        load_page(buffer.getvalue())


def test_api_includes_prose_verse_and_short_title_without_genre_labels(monkeypatch):
    blocks = [LayoutBlock([LayoutWord(text, x, 10, 80, 20, 95, (1, 1, 1))])
              for text, x in [('Текст прозы', 10), ('Текст стихов', 100), ('Заголовок', 190)]]
    blocks.append(LayoutBlock([LayoutWord('?!', 10, 100, 10, 10, 0, (1, 1, 1))], rejected='шум'))
    page = TextPageResult(blocks, np.full((300, 300), 255, np.uint8))
    monkeypatch.setattr(server, 'get_extractor', lambda: SimpleNamespace(extract=lambda _: page))
    response = TestClient(server.app).post('/api/ocr', files={'file': ('page.png', b'fixture')})
    body = response.json()
    assert body['text'] == 'Текст прозы\n\nТекст стихов\n\nЗаголовок'
    assert body['mode'] == 'all_text' and body['text_blocks'] == 3
    assert len(body['blocks']) == 3 and len(body['all_blocks']) == 4
    assert all(b['is_text'] for b in body['blocks'])
    assert 'prose_blocks' not in body and 'is_prose' not in body['blocks'][0]
    assert body['all_blocks'][-1]['rejected'] == 'шум'
    assert body['page_size'] == {'width': 300, 'height': 300}


def test_overlapping_reread_is_excluded_but_keeps_diagnostic_text():
    words = 'Это один и тот же абзац который прочитан повторно из пересекающихся рамок страницы журнала '
    text = words * 3
    full = TextBlock(30, 10, 600, 120, 10, 0.9, 10, True, text=text + ' Конец рассказа.')
    partial = TextBlock(0, 0, 400, 160, 10, 0.9, 10, True, text=text)
    ProseExtractor._remove_duplicate_text([full, partial])
    assert full.is_prose and not partial.is_prose
    assert partial.text == text and partial.rejected == 'повтор текста'


def test_repeated_words_without_physical_overlap_are_not_duplicates():
    text = 'Это повторяющийся текст но в другом месте страницы ' * 5
    blocks = [TextBlock(x, 0, 200, 200, 10, 0.9, 10, True, text=text) for x in [0, 300]]
    ProseExtractor._remove_duplicate_text(blocks)
    assert all(b.is_prose for b in blocks)
