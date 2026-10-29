"""Дополнительный разбор «проза / стихи» по рваности правого края (yunost/genre.py, /api/genre)."""
import os
import shutil

import pytest
from fastapi.testclient import TestClient

import server
from yunost.genre import GenreAnalyzer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
needs_tesseract = pytest.mark.skipif(shutil.which('tesseract') is None, reason='Tesseract не установлен')


def block(rows, width=600, height=20):
    """Блок из строк (текст, длина строки в пикселях) — как line_boxes в ответе /api/ocr."""
    return {'line_boxes': [dict(x=0, y=i * 30, right=right, bottom=i * 30 + height, text=text)
                           for i, (text, right) in enumerate(rows)]}


PROSE = block([('Скоро, в ночь на восьмое сентября, на перего-', 600),
               ('не Борисов — Орша, неподалеку от станции', 598),
               ('Крупки, загремели взорванные вагоны, гружен-', 601),
               ('ные орудиями и другой техникой. Вместе с под-', 599),
               ('рывниками участвовал в этой первой операции', 600),
               ('и Юра Озол, который ещё недавно учился в шко-', 600),
               ('ле и мечтал стать инженером на большом заводе', 602),
               ('в родном городе, где жили его отец и мать.', 400)])
VERSE = block([('Сегодня осень. И дождливой стуже', 420),
               ('Пора свобода месяца на два.', 330),
               ('Оканчивает жизнь, бросаясь в лужи,', 440),
               ('Безудержно нарядная листва.', 340),
               ('А рядом цветом вылинявшей ленты', 410),
               ('Отмечено течение реки.', 280)])


def test_prose_verse_and_short_blocks():
    result = GenreAnalyzer().analyze([PROSE, VERSE, block([('В. КОНСТАНТИНОВ', 200)]), block([('110', 30)])])
    kinds = [b['kind'] for b in result['blocks']]
    assert kinds == ['prose', 'verse', 'short', 'unknown']
    prose, verse = result['blocks'][:2]
    assert prose['raggedness'] < 0.35 < verse['raggedness']       # главный признак — рваность края
    assert result['counts'] == {'prose': 1, 'verse': 1, 'short': 1, 'unknown': 1}


def test_ragged_edge_alone_marks_verse():
    """Стихи без заглавных в начале строк (современная верстка) — по рваному краю."""
    lowercase = block([(t, r) for t, r in [('а за окном шумит река', 300), ('и снова дождь идёт,', 260),
                                           ('и где-то там, вдали', 230), ('кричит ночная птица', 280),
                                           ('и не спит никто', 200), ('и ждёт утра опять', 250)]])
    [verdict] = GenreAnalyzer().analyze([lowercase])['blocks']
    assert verdict['kind'] == 'verse' and verdict['raggedness'] > 0.35


def test_str_of_verdict():
    verdict = GenreAnalyzer().classify(0, VERSE)
    assert str(verdict).startswith('Блок 1: стихи (рваность края')


def test_api_validates_and_answers():
    client = TestClient(server.app)
    response = client.post('/api/genre', json={'blocks': [PROSE, VERSE]})
    assert response.status_code == 200
    assert [b['label'] for b in response.json()['blocks']] == ['проза', 'стихи']
    assert client.post('/api/genre', json={'blocks': 'нет'}).status_code == 422
    assert client.post('/api/genre', json={'blocks': [{}] * 501}).status_code == 422


@needs_tesseract
def test_real_page_after_ocr():
    """Страница с тремя колонками прозы и стихами внизу: после OCR проза и стихи разделяются."""
    with open(os.path.join(ROOT, 'data/norm/1957-10_p05.png'), 'rb') as fh:
        data = TestClient(server.app).post('/api/ocr', files={'file': ('p.png', fh, 'image/png')}).json()
    result = GenreAnalyzer().analyze(data['blocks'])
    texts = {kind: ' '.join(data['blocks'][b['index']]['text'] for b in result['blocks'] if b['kind'] == kind)
             for kind in ('prose', 'verse')}
    assert result['counts']['prose'] >= 3 and result['counts']['verse'] >= 1
    assert 'Борисовские' in texts['prose'] and 'Сегодня осень' in texts['verse']
    assert 'Сегодня осень' not in texts['prose']
