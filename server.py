"""
server.py — лендинг и API: загрузили фото страницы -> получили текст прозы.

Запуск:
    uvicorn server:app --reload
Лендинг: http://127.0.0.1:8000/     Документация API: http://127.0.0.1:8000/docs
"""
import base64
import logging
import os
import shutil

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from yunost import ProseExtractor, UnreadableImage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

MAX_UPLOAD = 25 * 1024 * 1024          # 25 МБ — с запасом для фото с телефона
HERE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Юность: проза со страницы журнала",
              description="Загрузите фото страницы — сервис найдёт блоки прозы и распознает их текст. "
                          "Стихи, заголовки и иллюстрации пропускаются. Файлы нигде не сохраняются.")
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")

_extractor = None


def get_extractor():
    """Движок создаём при первом запросе, чтобы сервер стартовал и без Tesseract."""
    global _extractor
    if _extractor is None:
        _extractor = ProseExtractor()
    return _extractor


@app.get("/", include_in_schema=False)
def landing():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/api/health")
def health():
    """Проверка, что Tesseract и русский язык установлены."""
    tesseract = shutil.which("tesseract")
    return {"status": "ok" if tesseract else "tesseract_missing", "tesseract": tesseract}


@app.post("/api/ocr")
def recognize(file: UploadFile = File(..., description="Фото или скан страницы: JPG, PNG, HEIC")):
    """Находит на странице блоки прозы и возвращает их текст и картинку с разметкой."""
    data = file.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Файл больше 25 МБ")
    if not data:
        raise HTTPException(400, "Файл пустой")
    try:
        result = get_extractor().extract(data)
    except UnreadableImage as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:                       # Tesseract не установлен, сбой OCR и т.п.
        logging.exception("OCR failed")
        raise HTTPException(500, "Не удалось распознать страницу. Проверьте, что установлен "
                                 "Tesseract с русским языком.") from exc

    overlay = base64.b64encode(result.overlay_png()).decode("ascii")
    return {
        "text": result.text,
        "prose_blocks": len(result.prose),
        "other_blocks": len(result.blocks) - len(result.prose),
        "blocks": [b.to_dict() for b in result.prose],
        "overlay": f"data:image/png;base64,{overlay}",
    }
