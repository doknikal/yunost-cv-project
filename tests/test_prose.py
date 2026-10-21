"""Legacy prose-only compatibility tests plus current all-text HTTP API tests."""
import glob
import os
import shutil
import sys

import cv2
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from yunost import BlockDetector, ProseExtractor            # noqa: E402
from yunost.blocks import TextBlock, reading_order           # noqa: E402
from yunost.ocr import OcrLine, join_lines, capital_start_ratio, word_quality  # noqa: E402

NORM = os.path.join(ROOT, "data", "norm")
PROSE_PAGE = os.path.join(NORM, "1957-10_p05.png")           # три колонки прозы + стихи в рамке
VERSE_PAGE = os.path.join(NORM, "1957-07_p02.png")           # только стихи
VERSE_PHOTO = os.path.join(NORM, "1965-02_p09.png")          # фото разворота со стихами
needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None,
                                     reason="Tesseract не установлен")


def read(path):
    with open(path, "rb") as fh:
        return fh.read()


# ---------- поиск блоков (без OCR) ----------

def test_prose_columns_are_candidates_before_ocr_verification():
    blocks = BlockDetector().detect(cv2.imread(PROSE_PAGE, cv2.IMREAD_GRAYSCALE))
    prose = [b for b in blocks if b.is_prose and b.y < 1300 and b.n_lines >= 40]
    assert len(prose) == 3
    assert [b.x for b in prose] == sorted(b.x for b in prose)    # слева направо
    assert all(b.n_lines >= 40 for b in prose)                   # три длинные колонки
    # A candidate below the separator can be a poem: only OCR verification
    # makes a final decision. The full-pipeline test below checks exclusion.


@pytest.mark.parametrize("path", [VERSE_PAGE, VERSE_PHOTO])
@needs_tesseract
def test_verse_pages_have_no_prose(path):
    assert not ProseExtractor().extract(read(path)).prose


def test_reading_order_columns_then_top_to_bottom():
    def blk(x, y):
        return TextBlock(x, y, 300, 200, 10, 1.0, 17)
    shuffled = [blk(700, 50), blk(0, 400), blk(0, 50), blk(700, 400)]
    assert [(b.x, b.y) for b in reading_order(shuffled)] == [(0, 50), (0, 400), (700, 50), (700, 400)]


# ---------- сборка текста ----------

def test_join_lines_merges_hyphenation_and_splits_paragraphs():
    lines = [OcrLine("первая строка сбро-", 0), OcrLine("шены на землю.", 0),
             OcrLine("Новый абзац начался", 40), OcrLine("и закончился.", 0)]
    assert join_lines(lines, indent=15) == ("первая строка сброшены на землю.\n\n"
                                            "Новый абзац начался и закончился.")


def test_text_quality_metrics():
    verse = [OcrLine(t, 0) for t in ("Ах, как пахнет июлем трава,", "Так, что кружится голова.")]
    prose = [OcrLine(t, 0) for t in ("будто в Борисовские леса", "сброшены две дивизии")]
    junk = [OcrLine(t, 0) for t in ("Е ЕЕ а - Н", "ва > ЛУ ^ 0")]
    assert capital_start_ratio(verse) == 1.0 and capital_start_ratio(prose) == 0.0
    assert word_quality(prose) > 0.6 > word_quality(junk)


# ---------- полный конвейер и API (нужен Tesseract) ----------

@needs_tesseract
def test_extractor_reads_prose_and_ignores_verse():
    ex = ProseExtractor()
    result = ex.extract(read(PROSE_PAGE))
    assert len(result.prose) == 3
    assert "Юра" in result.text and "Чуфистов" in result.text
    assert "осень" not in result.text                            # стихи Константинова не попали
    assert ex.extract(read(VERSE_PAGE)).text == ""


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    import server
    return TestClient(server.app)


def test_landing_and_health(client):
    assert client.get("/").status_code == 200
    assert "tesseract" in client.get("/api/health").json()


def test_api_rejects_bad_files(client):
    assert client.post("/api/ocr", files={"file": ("a.txt", b"not an image")}).status_code == 400
    assert client.post("/api/ocr", files={"file": ("a.png", b"")}).status_code == 400


@needs_tesseract
def test_api_returns_prose_and_verse_text_and_overlay(client):
    resp = client.post("/api/ocr", files={"file": ("page.png", read(PROSE_PAGE), "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert body['mode'] == 'all_text' and body['text_blocks'] > 3
    assert 'Чуфистов' in body['text'] and 'Сегодня осень' in body['text']
    assert body["overlay"].startswith("data:image/png;base64,")


@needs_tesseract
def test_api_opens_heic_photo(client):
    heic = sorted(glob.glob(os.path.join(ROOT, "1967-12", "1957", "1965-1", "*.heic")))
    if not heic:
        pytest.skip("нет HEIC-файлов в репозитории")
    resp = client.post("/api/ocr", files={"file": ("IMG.heic", read(heic[0]), "image/heic")})
    assert resp.status_code == 200
