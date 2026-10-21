"""Current app contract: text of any genre, short blocks and physical lineation."""
import io
import re
from pathlib import Path

import pytest
from PIL import Image

from yunost import TextExtractor
from yunost.layout import LayoutBlock, LayoutWord, layout_order
from yunost.ocr import TesseractEngine
from test_text_verification import russian_engine

ROOT = Path(__file__).resolve().parents[1]


def block(text, x=10, y=10, width=100, height=20, confidence=95):
    return LayoutBlock([LayoutWord(text, x, y, width, height, confidence, (1, 1, 1))])


def image_bytes():
    output = io.BytesIO()
    Image.new('L', (200, 200), 255).save(output, format='PNG')
    return output.getvalue()


def test_physical_lines_indentation_and_hyphens_are_not_prose_joined():
    b = LayoutBlock([LayoutWord(text, x, y, 30, 12, 95, (1, 1, i))
                     for i, (text, x, y) in enumerate([
                         ('До-', 10, 10), ('воль-', 10, 30), ('но!!', 10, 50),
                         ('К чертям!', 10, 70), ('Супермены!', 100, 90)])])
    lines = b.text.splitlines()
    assert lines[:4] == ['До-', 'воль-', 'но!!', 'К чертям!']
    assert lines[4].startswith(' ') and lines[4].strip() == 'Супермены!'


def test_sections_then_columns_then_footer_order():
    title = block('Заголовок', 10, 0, 300)
    left = block('Левая колонка', 10, 100, 120, 300)
    right = block('Правая колонка', 190, 100, 120, 300)
    footer = block('5', 10, 500, 10)
    assert layout_order([footer, right, title, left]) == [title, left, right, footer]


def test_sparse_fallback_finds_short_text_without_duplicate_words():
    class Engine:
        def read_layout(self, image, psm=3):
            if psm == 3:
                return [block('Проза'), block('Я', 10, 100, 15), block('2026', 10, 200)]
            if psm == 11:
                return [block('Проза'), block('Стихи', 10, 300)]
            return []

    result = TextExtractor(engine=Engine()).extract(image_bytes())
    assert len(result.text_blocks) == 4
    assert result.text.count('Проза') == 1
    assert 'Стихи' in result.text and 'Я' in result.text and '2026' in result.text


def test_blank_page_stays_empty():
    class EmptyEngine:
        def read_layout(self, image, psm=3):
            return []

    result = TextExtractor(engine=EmptyEngine()).extract(image_bytes())
    assert result.text == '' and result.text_blocks == []
    assert result.overlay_png().startswith(b'\x89PNG')


@pytest.fixture(scope='module')
def all_text_engine(russian_engine):
    # Fixture checks that the free local Russian weights are available.
    return TesseractEngine(scale=1)


def search_text(text):
    # Only comparison normalization: downloadable output keeps physical lines.
    text = re.sub(r'(?<=\w)-\s*\n\s*(?=[а-яё])', '', text)
    return re.sub(r'\s+', ' ', text).lower()


def test_user_staircase_page_all_columns_titles_short_lines_and_footer(all_text_engine):
    result = TextExtractor(engine=all_text_engine).extract((ROOT / 'data/norm/1967-12_p03.png').read_bytes())
    text = search_text(result.text)
    phrases = ['Не знаю', 'чего они этим добились', 'Мы очень понятливы',
               'РАЗГОВОР', 'СО СЛУЧАЙНЫМ ЗНАКОМЫМ', 'как дышит эта ночь',
               'МНОГОГОЛОСАЯ', 'ОБЫВАТЕЛЬЩИНА', 'Бунтуйте', 'РОБЕРТ РОЖДЕСТВЕНСКИЙ']
    positions = [text.index(p.lower()) for p in phrases]
    assert positions == sorted(positions)
    lines = [line.strip() for line in result.text.splitlines()]
    assert 'До-' in lines and 'воль-' in lines and 'но!!' in lines
    assert 'Люби' in lines and 'эту землю.' in lines
    indented = next(line for line in result.text.splitlines() if 'чего они этим добились' in line)
    assert indented.startswith(' ')
    assert text.count('чего они этим добились') == 1
    # No detected body block is allowed to join adjacent columns.
    assert all(b.box[2] < 450 for b in result.text_blocks if b.box[1] < 1760)


@pytest.mark.parametrize('filename,phrases', [
    ('1957-12_p06.png', ['В медсанбате все трое лежали рядом', 'Как Игорь был благодарен',
                         'Он бредит', 'Лев ОШАНИН', 'СЫН', 'Ты куришь',
                         'Как это было, помнишь', 'лучше не кури']),
    ('1967-10_p03.png', ['Если бы Кымытваль родилась', 'Вот что сказала Кымытваль',
                         'Она написала эти стихи', 'Горизонт', 'Случайный солнца луч',
                         'Беседу вел Владимир']),
    ('1957-10_p05.png', ['Чуфистов', 'День близился к концу', 'Сегодня осень',
                         'И ранит сердце', 'За мужество']),
])
def test_mixed_pages_include_both_genres_headings_and_endings(all_text_engine, filename, phrases):
    result = TextExtractor(engine=all_text_engine).extract((ROOT / 'data/norm' / filename).read_bytes())
    text = search_text(result.text)
    for phrase in phrases:
        assert phrase.lower() in text, phrase
    if filename == '1957-12_p06.png':
        assert text.index('как игорь') < text.index('сын') < text.index('ты куришь')
        assert text.index('опять спешу') < text.index('как это было')


def test_prose_columns_keep_three_column_reading_order(all_text_engine):
    result = TextExtractor(engine=all_text_engine).extract((ROOT / 'data/norm/1957-12_p03.png').read_bytes())
    text = search_text(result.text)
    anchors = ['Улетел', 'Лесник перестал ругаться', 'бугристую']
    positions = [text.index(anchor.lower()) for anchor in anchors]
    assert positions == sorted(positions)
