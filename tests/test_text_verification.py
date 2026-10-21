"""Conservative OCR-line verification, including real scan/photo regressions."""
import os
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from yunost.blocks import TextBlock  # noqa: E402
from yunost.ocr import OcrLine, TesseractEngine, join_lines, text_structure  # noqa: E402
from yunost.pipeline import ProseExtractor  # noqa: E402


def lines_with_geometry(texts, rights=None, lefts=None, stanza_every=None):
    lines = []
    top = 5
    for i, text in enumerate(texts):
        if stanza_every and i and i % stanza_every == 0:
            top += 30
        left = lefts[i] if lefts else 5
        right = rights[i] if rights else 600
        lines.append(OcrLine(text, left, right, top, top + 20))
        top += 30
    return lines


def verify(lines, flush=1.0):
    # No OCR dependency is needed to exercise the verifier.
    extractor = ProseExtractor(engine=object())
    block = TextBlock(0, 0, 620, 800, len(lines), flush, 20, is_prose=True)
    extractor._verify(block, join_lines(lines, 18), lines)
    return block


def test_optional_geometry_preserves_old_engine_contract():
    line = OcrLine("обычная строка текста", 4)
    assert line.right is None and line.top is None and line.bottom is None
    assert text_structure([line], 20).right_flush is None


def test_tesseract_records_line_extents_in_original_scale():
    class FakeTesseract:
        class Output:
            DICT = "dict"

        def image_to_data(self, image, **kwargs):
            return dict(text=["первое", "слово", "следующая"], conf=[90, 90, 90],
                        block_num=[1, 1, 1], par_num=[1, 1, 1], line_num=[1, 1, 2],
                        left=[12, 80, 10], top=[10, 12, 60],
                        width=[50, 40, 90], height=[30, 32, 28])

    import numpy as np

    engine = object.__new__(TesseractEngine)
    engine.timeout = 30
    engine.scale, engine.lang, engine._tess = 2, "rus", FakeTesseract()
    lines = engine.read_lines(np.full((50, 100), 255, dtype=np.uint8))
    assert lines == [OcrLine("первое слово", 6, 60, 5, 22),
                     OcrLine("следующая", 5, 50, 30, 44)]


def test_positive_wrapping_ignores_unreliable_detector_flush():
    lines = lines_with_geometry(["обычная длинная строка рассказа продолжается"] * 12)
    assert verify(lines, flush=0.0).is_prose


def test_ocr_raggedness_does_not_inherit_perfect_detector_flush():
    rights = [600, 390, 480, 290, 560, 360, 510, 280, 580, 420, 310, 450]
    lines = lines_with_geometry(["обычная строка продолжается дальше"] * 12, rights=rights)
    assert not verify(lines, flush=1.0).is_prose


def test_right_margin_metric_accounts_for_skew():
    lines = lines_with_geometry(["обычная строка продолжается дальше"] * 12)
    for line in lines:
        shift = int(0.12 * line.top)
        line.left += shift
        line.right += shift
    structure = text_structure(lines, 20)
    assert structure.right_flush == 1.0
    assert structure.deep_indent == 0.0


def test_perspective_can_make_left_and_right_margins_converge():
    lines = lines_with_geometry(["обычная строка продолжается дальше"] * 20)
    for i, line in enumerate(lines):
        line.right += 2 * i
    structure = text_structure(lines, 20)
    assert structure.right_flush == 1.0
    assert structure.deep_indent == 0.0
    assert verify(lines).is_prose


def test_repeated_deep_indents_reject_staircase_verse():
    texts = ["Над рекою туманы плывут", "поднимаются тихо над садом"] * 8
    lines = lines_with_geometry(texts, lefts=[5, 200] * 8)
    block = verify(lines)
    assert not block.is_prose and block.rejected == "стихи"
    assert "туманы" in block.text  # Diagnostic OCR must not be discarded.


def test_normal_paragraph_indentation_is_not_staircase_verse():
    texts = ["обычная длинная строка рассказа продолжается"] * 12
    lines = lines_with_geometry(texts, lefts=[35 if i % 4 == 0 else 5 for i in range(12)])
    assert text_structure(lines, 20).deep_indent == 0.0
    assert verify(lines).is_prose


def test_capitalized_verse_is_rejected_even_with_regular_right_margin():
    lines = lines_with_geometry(["Снова над рекою туманы плывут"] * 12)
    block = verify(lines)
    assert not block.is_prose and block.rejected == "стихи"


def test_stanza_gaps_are_not_positive_prose_evidence():
    texts = ["над рекою туманы плывут далеко"] * 12
    lines = lines_with_geometry(texts, stanza_every=4)
    assert text_structure(lines, 20).stanza_breaks > 0.1
    assert not verify(lines).is_prose


def test_word_hyphenation_supports_short_or_ragged_prose():
    texts = ["первая строка рассказа продол-", "жается следующими словами повествования",
             "герои рассказа вновь возвра-", "щаются домой после путешествия",
             "продолжение обычного связного рассказа"]
    lines = lines_with_geometry(texts, rights=[600, 450, 520, 410, 350])
    assert verify(lines, flush=0.0).is_prose
    legacy_lines = [OcrLine(line.text, line.left) for line in lines]
    assert verify(legacy_lines, flush=0.0).is_prose


def test_five_similarly_long_lowercase_lines_are_ambiguous_not_prose():
    lines = lines_with_geometry(["над рекою туманы плывут далеко"] * 5)
    block = verify(lines)
    assert not block.is_prose and block.rejected == "мало признаков прозы"


def test_short_sentences_alone_do_not_label_text_as_verse():
    lines = lines_with_geometry(["наступило прекрасное утро."] * 12)
    block = verify(lines)
    assert not block.is_prose and block.rejected == "мало признаков прозы"


def test_junk_and_insufficient_ocr_lines_are_rejected():
    assert verify([OcrLine("Е ЕЕ а - Н", 0)] * 12).rejected == "не текст"
    assert verify([OcrLine("обычная длинная строка рассказа", 0)] * 4).rejected == "мало строк"


@pytest.fixture(scope="module")
def russian_engine():
    if shutil.which("tesseract") is None:
        pytest.skip("Tesseract is not installed")
    fast = ROOT / ".venv/tessdata/fast"
    old = os.environ.get("TESSDATA_PREFIX")
    if (fast / "rus.traineddata").exists():
        os.environ["TESSDATA_PREFIX"] = str(fast)
    import pytesseract
    if "rus" not in pytesseract.get_languages(config=""):
        if old is None:
            os.environ.pop("TESSDATA_PREFIX", None)
        else:
            os.environ["TESSDATA_PREFIX"] = old
        pytest.skip("Russian Tesseract data is not installed")
    try:
        yield TesseractEngine()
    finally:
        if old is None:
            os.environ.pop("TESSDATA_PREFIX", None)
        else:
            os.environ["TESSDATA_PREFIX"] = old


@pytest.mark.parametrize("filename, expected_blocks, phrase", [
    ("1957-10_p05.png", 3, "Юра"),
    ("1963-12_p07.png", 2, "Эльги"),
    # Column recovery joins fragments, including the formerly missing endings.
    ("1967-10_p03.png", 3, "Кымытваль"),
])
def test_real_prose_scan_and_photos(russian_engine, filename, expected_blocks, phrase):
    result = ProseExtractor(engine=russian_engine).extract((ROOT / "data/norm" / filename).read_bytes())
    assert len(result.prose) == expected_blocks
    assert phrase in result.text


@pytest.mark.parametrize('filename,phrase', [
    ('1957-06_p05.png', 'традиций'),
    ('1965-02_p10.png', 'пословицах'),
    ('1957-07_p04.png', 'Эммет')])
def test_editorial_introductions_are_not_lost_on_poetry_pages(russian_engine, filename, phrase):
    result = ProseExtractor(engine=russian_engine).extract((ROOT / 'data/norm' / filename).read_bytes())
    assert phrase.lower() in result.text.lower()
    assert len(result.prose) >= 1


@pytest.mark.parametrize("filename", [
    "1957-02_p06.png", "1957-07_p02.png", "1957-11_p02.png", "1957-11_p05.png",
    "1963-12_p04.png", "1965-01_p04.png", "1965-02_p01.png",
    "1965-02_p03.png", "1965-02_p04.png", "1965-02_p09.png",
])
def test_real_verse_scan_and_photos_have_no_prose(russian_engine, filename):
    result = ProseExtractor(engine=russian_engine).extract((ROOT / "data/norm" / filename).read_bytes())
    assert not result.prose
    assert result.text == ""
