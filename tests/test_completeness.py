"""Region/text regression tests: page presence alone does not measure completeness."""
import json
from pathlib import Path

import numpy as np
import pytest

from yunost.blocks import TextBlock
from yunost.pipeline import PageResult, ProseExtractor
from yunost.recovery import column_extensions
from test_text_verification import russian_engine

ROOT = Path(__file__).resolve().parents[1]


def sample_column():
    mask = np.zeros((600, 1000), np.uint8)
    for y in [100, 125, 150, 175, 200, 225, 260, 285]:
        # Disconnected word components: entire column evidence matters.
        for x in range(100, 460, 40):
            mask[y:y + 12, x:x + 24] = 255
    anchor = TextBlock(100, 100, 400, 137, 6, 1.0, 15, True)
    return mask, anchor


def test_short_lines_recovered_only_with_confirmed_prose_anchor():
    mask, anchor = sample_column()
    assert column_extensions(mask, []) == []
    proposals = column_extensions(mask, [anchor])
    assert len(proposals) == 1
    assert proposals[0].y + proposals[0].h == 297


def test_large_gap_stops_recovery():
    mask, anchor = sample_column()
    mask[260:300] = 0
    mask[350:362, 100:480] = 255
    assert column_extensions(mask, [anchor]) == []


def test_centred_author_is_not_prepended_despite_left_margin_speckle():
    mask, anchor = sample_column()
    mask[70:82, 360:480] = 255
    mask[70:72, 100:102] = 255
    assert all(b.y == 100 for b in column_extensions(mask, [anchor]))


def test_rejected_long_poem_is_a_recovery_barrier():
    mask, anchor = sample_column()
    poem = TextBlock(100, 255, 400, 250, 10, 0.2, 15, False, 'стихи',
                     raw_text='Текст стихов', ocr_metrics={'lines': 10})
    assert column_extensions(mask, [anchor, poem]) == []


def test_rejected_spanning_box_does_not_change_prose_column_order():
    left = TextBlock(10, 15, 100, 100, 6, 1, 15, True, text='Левая колонка')
    right = TextBlock(130, 10, 100, 100, 6, 1, 15, True, text='Правая колонка')
    artwork = TextBlock(0, 130, 250, 100, 8, 0, 15)
    page = PageResult([right, left, artwork], np.zeros((250, 250), np.uint8))
    assert page.text == 'Левая колонка\n\nПравая колонка'


@pytest.mark.parametrize('filename,phrases,excluded', [
    ('1967-10_p03.png', [
        'Вот что сказала Кымытваль', 'Мы шли по улице',
        'На Чукотке бывает значительно', 'там одеваются теплее',
        'Потом, улыбнувшись, она закончила', 'напишу вечером',
        'Она написала эти стихи', 'В переводе Леонида Завальнюка',
        'Главный герой', 'Я рада, когда читатели пишут мне'],
     ['Горизонт — это то', 'Случайный солнца луч']),
    ('1957-12_p06.png', [
        'прошептал Игорь', 'Вы что-то сказали, дорогой', 'и тогда забуду',
        'Он бредит', 'Но он не бредил', 'Уже искупил', 'Но забудет ли',
        'Выбравшись вслед за командиром', 'В медсанбате все трое лежали рядом'],
     ['Мы, опоздав к семнадцатому году', 'Ну как там чайки']),
])
def test_missing_passages_return_without_verse(russian_engine, filename, phrases, excluded):
    result = ProseExtractor(engine=russian_engine).extract((ROOT / 'data/norm' / filename).read_bytes())
    for phrase in phrases:
        assert phrase.lower() in result.text.lower(), phrase
    for phrase in excluded:
        assert phrase.lower() not in result.text.lower(), phrase
    labels = json.loads((ROOT / 'docs/ocr_evaluation/completeness_labels.json').read_text())
    page = next(p for p in labels['pages'] if p['file'] == filename)
    covered = np.zeros(result.gray.shape, np.uint8)
    for block in result.prose:
        covered[block.y:block.y + block.h, block.x:block.x + block.w] = 1
    for passage in page['missing_passages']:
        x0, y0, x1, y1 = passage['bounds']
        # References enclose whitespace as well as glyphs; require spatial
        # coverage too, so a similar phrase elsewhere cannot pass this test.
        assert covered[y0:y1, x0:x1].mean() >= 0.80, passage['id']
    # Confirm left-to-right reading, not just a bag of recovered phrases.
    if filename == '1957-12_p06.png':
        assert result.text.index('В медсанбате') < result.text.index('Как Игорь был благодарен')
