"""
pipeline.py — от файла с фото страницы до текста прозы.

    файл -> страница (нужной высоты, в оттенках серого)
         -> BlockDetector: блоки текста, среди них проза
         -> OcrEngine: читает только блоки-прозу
         -> PageResult: текст, блоки и картинка с разметкой
"""
import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

import cvpipe
from .blocks import BlockDetector, ProseSettings, TextBlock
from .decorators import timed
from .ocr import OcrEngine, TesseractEngine, capital_start_ratio, cyrillic_ratio, word_quality

try:                                   # фото с iPhone приходят в формате HEIC
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:                    # без pillow-heif HEIC просто не откроется
    pass

Image.MAX_IMAGE_PIXELS = 80_000_000    # защита от «бомб»: слишком большие картинки не открываем
TARGET_HEIGHT = 2000                   # все пороги cvpipe подобраны для такой высоты
PROSE_COLOR = (60, 160, 0)             # BGR
OTHER_COLOR = (150, 150, 150)


class UnreadableImage(ValueError):
    """Файл не удалось открыть как картинку."""


def load_page(data):
    """Байты файла -> страница в оттенках серого высотой TARGET_HEIGHT."""
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("L")   # учитываем поворот с телефона
    except Exception as exc:
        raise UnreadableImage("Не удалось открыть файл как изображение") from exc
    width = round(img.width * TARGET_HEIGHT / img.height)
    return np.array(img.resize((width, TARGET_HEIGHT), Image.LANCZOS))


@dataclass
class PageResult:
    blocks: list
    gray: np.ndarray

    @property
    def prose(self):
        return [b for b in self.blocks if b.is_prose]

    @property
    def text(self):
        return "\n\n".join(b.text for b in self.prose if b.text)

    def overlay_png(self):
        """Страница с рамками: зелёные — проза (прочитана), серые — всё остальное."""
        vis = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
        for number, b in enumerate(self.prose, start=1):
            cv2.rectangle(vis, (b.x, b.y), (b.x + b.w, b.y + b.h), PROSE_COLOR, 6)
            cv2.putText(vis, str(number), (b.x + 10, b.y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, PROSE_COLOR, 6)
        for b in self.blocks:
            if not b.is_prose:
                cv2.rectangle(vis, (b.x, b.y), (b.x + b.w, b.y + b.h), OTHER_COLOR, 3)
        ok, png = cv2.imencode(".png", cv2.resize(vis, None, fx=0.5, fy=0.5,
                                                  interpolation=cv2.INTER_AREA))
        return png.tobytes()


class ProseExtractor:
    """Главный объект: достаёт прозу со страницы журнала."""

    def __init__(self, engine=None, settings=None):
        self.engine = engine or TesseractEngine()
        self.settings = settings or ProseSettings()
        self.detector = BlockDetector(self.settings)

    @timed
    def extract(self, data):
        gray = load_page(data)
        blocks = self.detector.detect(gray)
        flat = cvpipe.normalize_illumination(gray)       # ровный фон — лучше для OCR
        pad = 6
        for b in blocks:
            if not b.is_prose:
                continue
            crop = flat[max(0, b.y - pad):b.y + b.h + pad,
                        max(0, b.x - pad):b.x + b.w + pad]
            text, lines = self.engine.read_block(crop, b.line_h)
            self._verify(b, text, lines)
        return PageResult(blocks, gray)

    def _verify(self, block, text, lines):
        """Вторая проверка по прочитанному тексту: отсеивает стихи с ровными строками
        и блоки, где вместо текста ноты или рисунок."""
        if (cyrillic_ratio(lines) < self.settings.min_cyrillic
                or word_quality(lines) < self.settings.min_word_quality):
            block.is_prose, block.rejected = False, "не текст"
        elif capital_start_ratio(lines) > self.settings.max_capital_start:
            block.is_prose, block.rejected = False, "стихи"
        else:
            block.text = text
