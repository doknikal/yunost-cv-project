"""
pipeline.py — старый конвейер только прозы (совместимость и исторические оценки).
Приложение использует жанрово-нейтральный TextExtractor из text.py.

    файл -> страница (нужной высоты, в оттенках серого)
         -> BlockDetector: кандидаты по геометрии
         -> подготовка и OcrEngine: физические строки кандидатов
         -> проверка структуры: подтверждённая проза
         -> PageResult: текст, блоки и картинка с разметкой
"""
import io
import re
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher

import cv2
import numpy as np
from PIL import Image, ImageOps

import cvpipe
from .blocks import BlockDetector, ProseSettings, TextBlock, reading_order
from .decorators import timed
from .preprocessing import prepare_text
from .recovery import column_extensions
from .ocr import OcrEngine, TesseractEngine, capital_start_ratio, cyrillic_ratio, word_quality, text_structure

try:                                   # фото с iPhone приходят в формате HEIC
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:                    # без pillow-heif HEIC просто не откроется
    pass

Image.MAX_IMAGE_PIXELS = 80_000_000    # защита от «бомб»: слишком большие картинки не открываем
TARGET_HEIGHT = 2000                   # все пороги cvpipe подобраны для такой высоты
MAX_PAGE_WIDTH = 4000                  # ограничиваем память на широких разворотах
PROSE_COLOR = (60, 160, 0)             # BGR
OTHER_COLOR = (150, 150, 150)


class UnreadableImage(ValueError):
    """Файл не удалось открыть как картинку."""


def load_page(data):
    """Decode one page; apply EXIF and composite transparency onto white paper."""
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.width * source.height > Image.MAX_IMAGE_PIXELS:
                raise UnreadableImage("Изображение больше 80 мегапикселей")
            if getattr(source, 'n_frames', 1) != 1:
                raise UnreadableImage("Загрузите одну страницу, не анимацию или многостраничный файл")
            if min(source.size) < 64 or not 0.15 <= source.width / source.height <= 6:
                raise UnreadableImage("Загрузите целую страницу: изображение слишком узкое или маленькое")
            img = ImageOps.exif_transpose(source)
            if 'A' in img.getbands() or 'transparency' in img.info:
                rgba = img.convert('RGBA')
                paper = Image.new('RGBA', rgba.size, 'white')
                img = Image.alpha_composite(paper, rgba)
            img = img.convert('L')
            scale = min(TARGET_HEIGHT / img.height, MAX_PAGE_WIDTH / img.width)
            size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
            return np.array(img.resize(size, Image.LANCZOS))
    except UnreadableImage:
        raise
    except Exception as exc:
        raise UnreadableImage("Не удалось открыть файл как изображение") from exc


@dataclass
class PageResult:
    blocks: list
    gray: np.ndarray

    @property
    def prose(self):
        # Rejected wide artwork/verse must not bridge two genuine columns.
        return reading_order([b for b in self.blocks if b.is_prose])

    @property
    def text(self):
        return "\n\n".join(b.text for b in self.prose if b.text)

    def overlay_png(self):
        """Страница с рамками: зелёные — проза (прочитана), серые — всё остальное."""
        vis = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
        for number, b in enumerate(self.blocks, start=1):
            color = PROSE_COLOR if b.is_prose else OTHER_COLOR
            cv2.rectangle(vis, (b.x, b.y), (b.x + b.w, b.y + b.h), color, 6 if b.is_prose else 3)
            cv2.putText(vis, str(number), (b.x + 5, b.y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
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
        for b in blocks:
            if not b.is_prose:
                continue
            text, lines = self._read(b, flat)
            self._verify(b, text, lines)
        for b in column_extensions(self.detector.text_mask(gray), blocks):
            text, lines = self._read(b, flat)
            self._verify(b, text, lines)
            b.ocr_metrics['column_recovery'] = True
            # A prose ending made of short dialogue sentences can lower the
            # overall flush/hyphen ratio. Require an actual, well-retained prose
            # anchor rather than weakening the standalone short-block gate.
            anchors = [a for a in blocks if a.is_prose and
                       b.overlap_with(a) >= 0.95 * a.area]
            words = re.findall(r'[а-яё]+', text.lower())
            retained = []
            for anchor in anchors:
                original = re.findall(r'[а-яё]+', anchor.text.lower())
                matches = SequenceMatcher(None, original, words, autojunk=False)
                retained.append(sum(m.size for m in matches.get_matching_blocks()) /
                                max(1, len(original)))
            b.ocr_metrics['anchor_word_coverage'] = max(retained, default=0.0)
            if (b.rejected == 'мало признаков прозы' and retained and max(retained) >= 0.90
                    and b.ocr_metrics['continuation'] >= self.settings.min_continuation
                    and b.ocr_metrics['hyphenated'] * (b.ocr_metrics['lines'] - 1) >= 2):
                b.is_prose, b.rejected = True, ''
            blocks.append(b)
        blocks = reading_order(blocks)
        self._remove_duplicate_text(blocks)
        return PageResult(blocks, gray)

    def _read(self, block, flat):
        pad = 6
        crop = flat[max(0, block.y - pad):block.y + block.h + pad,
                    max(0, block.x - pad):block.x + block.w + pad]
        crop, block.deskew_angle, block.noise_level = prepare_text(crop)
        text, lines = self.engine.read_block(crop, block.line_h)
        block.text = text
        block.raw_text = "\n".join(line.text for line in lines)
        block.ocr_metrics = asdict(text_structure(lines, block.line_h))
        block.ocr_metrics['capital_start'] = capital_start_ratio(lines)
        return text, lines

    @staticmethod
    def _remove_duplicate_text(blocks):
        """Drop overlapping rereads of the same passage, not repeated prose elsewhere.

        A frame may create a second partial crop of a wide introduction. Geometry
        alone cannot remove it because neither rectangle contains the other.
        Require substantial physical overlap AND matching word sequences; keep
        the fuller reading, retaining the partial OCR in diagnostics.
        """
        accepted = [b for b in blocks if b.is_prose]
        tokens = {id(b): re.findall(r"[а-яё]+", b.text.lower()) for b in accepted}
        accepted.sort(key=lambda b: len(tokens[id(b)]), reverse=True)
        kept = []
        for block in accepted:
            words = tokens[id(block)]
            duplicate = False
            if len(words) >= 20:
                for other in kept:
                    if block.overlap_with(other) < 0.5 * min(block.area, other.area):
                        continue
                    matches = SequenceMatcher(None, words, tokens[id(other)], autojunk=False)
                    if sum(m.size for m in matches.get_matching_blocks()) >= 0.80 * len(words):
                        duplicate = True
                        break
            if duplicate:
                block.is_prose, block.rejected = False, "повтор текста"
            else:
                kept.append(block)

    def _verify(self, block, text, lines):
        """Require positive prose evidence; detector candidates can include poems."""
        block.text = text  # Rejected OCR remains available for diagnostics only.
        is_prose, reason = prose_decision(lines, block.line_h, self.settings, fallback_flush=block.flush)
        block.is_prose, block.rejected = is_prose, reason


def prose_decision(lines, line_h, settings, fallback_flush=None):
    """Решение «проза или нет» по физическим строкам OCR: (is_prose, причина отказа).

    Главный признак — ровность правого края (обратная сторона рваности); рядом —
    переносы слов, продолжение фразы на следующей строке, строки с заглавной буквы,
    отступы «лесенкой» и пустые строки между строфами. Используется и старым
    конвейером прозы, и дополнительным разбором «проза / стихи» (genre.py).
    """
    s = settings
    structure = text_structure(lines, line_h)
    if (cyrillic_ratio(lines) < s.min_cyrillic
            or word_quality(lines) < s.min_word_quality):
        return False, "не текст"
    if structure.lines < s.min_lines:
        return False, "мало строк"
    if (capital_start_ratio(lines) > s.max_capital_start
            or structure.deep_indent > s.max_deep_indent):
        return False, "стихи"
    # Word hyphenation and lowercase continuations are independent of
    # the unreliable pixel-band flushness of a tilted candidate. For
    # custom OCR engines without coordinates, use that score only as
    # a fallback, never as sufficient evidence on its own.
    flush = structure.right_flush
    if flush is None:
        flush = fallback_flush
    wrapping = structure.continuation >= s.min_continuation
    hyphenated = (structure.hyphenated >= s.min_hyphenated
                  and structure.hyphenated * (structure.lines - 1) >= 2)
    # A single word break can support a short prose paragraph, but
    # only with near-continuous wrapping AND a very consistent margin.
    short_wrapped = (structure.hyphenated > 0 and flush is not None and flush >= 0.85
                     and structure.continuation >= 0.80
                     and structure.stanza_breaks < 0.10)
    # A handful of similarly long poem lines can look justified by
    # chance. Without word hyphenation, require a larger sample.
    regular = (structure.lines >= s.min_unhyphenated_lines and flush is not None
               and flush >= s.min_ocr_flush and structure.stanza_breaks < 0.10)
    if wrapping and (regular or hyphenated or short_wrapped):
        return True, ""
    return False, "мало признаков прозы"
