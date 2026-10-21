"""
server.py — лендинг и API: загрузили фото страницы -> получили весь найденный текст.

Запуск:
    uvicorn server:app --reload
Лендинг: http://127.0.0.1:8000/     Документация API: http://127.0.0.1:8000/docs
"""
import base64
import functools
import logging
import os
import shutil
import sqlite3
import subprocess
import threading
from typing import Literal

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from starlette.datastructures import Headers
from starlette.formparsers import MultiPartException
from starlette.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from yunost import TextExtractor, UnreadableImage
from yunost.ocr import OCRTimeout
from yunost.analytics import AnalyticsService, Metric
from yunost.storage import CorpusFilter, CorpusRepository, default_database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

MAX_UPLOAD = 25 * 1024 * 1024          # 25 МБ — с запасом для фото с телефона
HERE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Юность: текст со страницы журнала",
              description="Сервис распознаёт блоки текста без отбора по жанру: прозу, стихи, "
                          "заголовки и короткие фрагменты. Сохраняет строки и отступы. "
                          "Загрузки удаляются после обработки.")
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")

_extractor = None
_ocr_slot = threading.BoundedSemaphore(1)
_analytics = None
_analytics_lock = threading.Lock()


class UploadSizeLimit:
    """Bound multipart bodies, including chunked uploads, before parsing to disk."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['path'] != '/api/ocr':
            return await self.app(scope, receive, send)
        limit = MAX_UPLOAD + 64 * 1024  # multipart headers/boundary overhead
        length = Headers(scope=scope).get('content-length')
        if length is not None:
            try:
                size = int(length)
                if size < 0:
                    raise ValueError
            except ValueError:
                return await JSONResponse({'detail': 'Некорректный размер запроса'}, 400)(scope, receive, send)
            if size > limit:
                return await JSONResponse({'detail': 'Файл больше 25 МБ'}, 413)(scope, receive, send)
        received, exceeded = 0, False

        async def bounded_receive():
            nonlocal received, exceeded
            message = await receive()
            if message['type'] == 'http.request':
                received += len(message.get('body', b''))
                if received > limit:
                    exceeded = True
                    # MultipartParser closes its temporary files on this exception.
                    raise MultiPartException('Файл больше 25 МБ')
            return message

        async def bounded_send(message):
            if exceeded and message['type'] == 'http.response.start':
                message = dict(message, status=413)
            await send(message)

        await self.app(scope, bounded_receive, bounded_send)


app.add_middleware(UploadSizeLimit)


def get_extractor():
    """Движок создаём при первом запросе, чтобы сервер стартовал и без Tesseract."""
    global _extractor
    if _extractor is None:
        _extractor = TextExtractor()
    return _extractor


@app.get("/", include_in_schema=False)
def landing():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


def get_analytics():
    """Lazy import only for an empty research DB; uploads are never persisted."""
    global _analytics
    with _analytics_lock:
        if _analytics is None:
            try:
                repository = CorpusRepository(default_database())
                repository.ensure_seeded(os.path.join(HERE, 'data/manifest.csv'),
                                         os.path.join(HERE, 'data/features.csv'))
                _analytics = AnalyticsService(repository)
            except (OSError, sqlite3.Error, ValueError, KeyError) as exc:
                logging.exception('Research corpus database unavailable')
                raise HTTPException(503, 'Не удалось открыть базу корпуса. Проверьте права на папку data '
                                    'и выполните import_corpus.py.') from exc
    return _analytics


def corpus_filters(year_min: int = Query(1800, ge=1800, le=2200),
                   year_max: int = Query(2200, ge=1800, le=2200),
                   source: Literal['all', 'scan', 'photo'] = 'all'):
    try:
        return CorpusFilter(year_min, year_max, source)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get('/analytics', include_in_schema=False)
def analytics_page():
    return FileResponse(os.path.join(HERE, 'static/analytics.html'))


@functools.lru_cache(maxsize=1)
def plotly_js():
    """plotly.js из установленного пакета plotly (offline, без CDN); читаем один раз — кэш."""
    from plotly.offline import get_plotlyjs
    return get_plotlyjs()


@app.get('/vendor/plotly.min.js', include_in_schema=False)
def plotly_bundle():
    return Response(plotly_js(), media_type='application/javascript',
                    headers={'Cache-Control': 'public, max-age=86400'})


@app.get('/api/analytics/options')
def analytics_options():
    return get_analytics().options()


@app.get('/api/analytics/histogram')
def analytics_histogram(metric: Metric = 'raggedness', bins: int = Query(12, ge=2, le=60),
                        selection: CorpusFilter = Depends(corpus_filters)):
    return get_analytics().histogram(metric, bins, selection)


@app.get('/api/analytics/trend')
def analytics_trend(metric: Metric = 'raggedness', aggregation: Literal['mean', 'median'] = 'median',
                    selection: CorpusFilter = Depends(corpus_filters)):
    return get_analytics().trend(metric, aggregation, selection)


@app.get('/api/analytics/scatter')
def analytics_scatter(x_metric: Metric = 'n_columns', y_metric: Metric = 'raggedness',
                      selection: CorpusFilter = Depends(corpus_filters)):
    return get_analytics().scatter(x_metric, y_metric, selection)


@app.get("/api/health")
@app.get("/health", include_in_schema=False)
def health():
    """Проверка, что Tesseract и русский язык установлены."""
    tesseract = shutil.which("tesseract")
    languages = []
    status = "tesseract_missing"
    if tesseract:
        try:
            result = subprocess.run([tesseract, "--list-langs"], capture_output=True,
                                    text=True, timeout=5, check=True)
            languages = result.stdout.splitlines()[1:]
            status = "ok" if "rus" in languages else "russian_missing"
        except (subprocess.SubprocessError, OSError):
            status = "tesseract_error"
    return {"status": status, "tesseract": tesseract, "languages": languages}


@app.post("/api/ocr")
def recognize(file: UploadFile = File(..., description="Фото или скан страницы: JPG, PNG, HEIC")):
    """Находит все текстовые блоки и возвращает текст, координаты и разметку."""
    data = file.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Файл больше 25 МБ")
    if not data:
        raise HTTPException(400, "Файл пустой")
    if not _ocr_slot.acquire(blocking=False):
        raise HTTPException(429, "Другая страница ещё распознаётся. Повторите через несколько секунд.",
                            headers={'Retry-After': '3'})
    try:
        result = get_extractor().extract(data)
        return page_response(result)
    except UnreadableImage as exc:
        raise HTTPException(400, str(exc))
    except OCRTimeout as exc:
        raise HTTPException(504, "Распознавание заняло слишком много времени. Попробуйте более чёткий скан.") from exc
    except Exception as exc:                       # Tesseract не установлен, сбой OCR и т.п.
        logging.exception("OCR failed")
        raise HTTPException(500, "Не удалось распознать страницу. Проверьте, что установлен "
                                  "Tesseract с русским языком.") from exc
    finally:
        _ocr_slot.release()


def page_response(result):
    overlay = base64.b64encode(result.overlay_png()).decode("ascii")
    return {
        "text": result.text,
        "text_blocks": len(result),
        "other_blocks": len(result.blocks) - len(result),
        "blocks": [b.to_dict() for b in result],
        "first_block": result[0].to_dict() if len(result) else None,
        "all_blocks": [b.to_dict() for b in result.blocks],
        "engine": "tesseract",
        "mode": "all_text",
        "page_size": {"width": result.gray.shape[1], "height": result.gray.shape[0]},
        "overlay": f"data:image/png;base64,{overlay}",
    }
