"""
ocr.py — движки распознавания текста.

Движок получает картинку ОДНОГО блока текста и возвращает строки с отступами.
Чтобы подключить другой OCR (например, PaddleOCR), достаточно написать
наследника OcrEngine с методом read_lines().
"""
from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import numpy as np


class OCRTimeout(TimeoutError):
    """An OCR process or the complete page exceeded its time budget."""


_deadline = ContextVar('ocr_deadline', default=None)


@contextmanager
def ocr_budget(seconds=90):
    token = _deadline.set(time.monotonic() + seconds)
    try:
        yield
    finally:
        _deadline.reset(token)


@dataclass
class OcrLine:
    text: str
    left: int          # отступ слева в пикселях (нужен, чтобы найти начало абзаца)
    right: int | None = None
    top: int | None = None
    bottom: int | None = None


class OcrEngine(ABC):
    """Общий интерфейс: блок текста (numpy-картинка) -> текст. Абстрактный класс (лекция 2)."""

    name = "base"

    @abstractmethod
    def read_lines(self, image):
        """Возвращает список OcrLine по строкам блока."""

    def read_block(self, image, line_h):
        """Читает блок. Возвращает (текст, строки): текст — абзацы разделены пустой
        строкой, переносы склеены."""
        lines = [ln for ln in self.read_lines(image) if ln.text.strip()]
        return join_lines(lines, indent=0.9 * line_h), lines


class TesseractEngine(OcrEngine):
    """Tesseract (бесплатный, работает на обычном процессоре).

    Наследник OcrEngine и одновременно «Адаптер» из лекции 5: приводит интерфейс библиотеки
    pytesseract (image_to_data) к нашему интерфейсу read_lines()/read_layout()."""

    name = "tesseract"

    def __init__(self, lang="rus", scale=1.5, timeout=30):
        import pytesseract                         # импорт здесь, чтобы тесты без OCR работали
        self._tess = pytesseract
        self.lang = lang
        self.scale = scale
        self.timeout = timeout

    def _data(self, image, psm):
        deadline = _deadline.get()
        timeout = self.timeout if deadline is None else min(self.timeout, deadline - time.monotonic())
        if timeout <= 0:
            raise OCRTimeout('Превышено время распознавания страницы')
        try:
            return self._tess.image_to_data(image, lang=self.lang, config=f'--psm {psm}',
                                            output_type=self._tess.Output.DICT, timeout=timeout)
        except RuntimeError as exc:
            if 'timeout' in str(exc).lower():
                raise OCRTimeout('Превышено время распознавания страницы') from exc
            raise

    def read_lines(self, image):
        import cv2
        if self.scale != 1:
            image = cv2.resize(image, None, fx=self.scale, fy=self.scale,
                               interpolation=cv2.INTER_CUBIC)
        data = self._data(image, 6)
        lines = {}
        for i, word in enumerate(data["text"]):
            if not word.strip() or float(data["conf"][i]) < 0:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            entry = lines.setdefault(key, {"words": [], "left": data["left"][i],
                                           "right": 0, "top": data["top"][i], "bottom": 0})
            entry["words"].append(word)
            entry["left"] = min(entry["left"], data["left"][i])
            entry["right"] = max(entry["right"], data["left"][i] + data["width"][i])
            entry["top"] = min(entry["top"], data["top"][i])
            entry["bottom"] = max(entry["bottom"], data["top"][i] + data["height"][i])
        ordered = sorted(lines.items(), key=lambda kv: kv[0])
        return [OcrLine(" ".join(v["words"]), int(v["left"] / self.scale),
                        int(v["right"] / self.scale), int(v["top"] / self.scale),
                        int(v["bottom"] / self.scale))
                for _, v in ordered]

    def read_layout(self, image, psm=3):
        """Detect ALL page text, including headings and one-word physical lines.

        Return block/line/word geometry in original image pixels. PSM 3 finds
        columns; PSM 11 is a complementary sparse-text pass, not a genre filter.
        """
        import cv2
        from .layout import LayoutBlock, LayoutWord

        if self.scale != 1:
            image = cv2.resize(image, None, fx=self.scale, fy=self.scale,
                               interpolation=cv2.INTER_CUBIC)
        data = self._data(image, psm)
        blocks = {}
        for i, text in enumerate(data['text']):
            if not text.strip() or float(data['conf'][i]) < 0:
                continue
            block_id = data['block_num'][i]
            coords = [round(data[key][i] / self.scale) for key in ('left', 'top', 'width', 'height')]
            word = LayoutWord(text.strip(), *coords, float(data['conf'][i]),
                              (block_id, data['par_num'][i], data['line_num'][i]))
            blocks.setdefault(block_id, []).append(word)
        return [LayoutBlock(words, 'sparse' if psm == 11 else 'page')
                for words in blocks.values()]


def join_lines(lines, indent):
    """Собирает строки в текст: красная строка (отступ) начинает новый абзац,
    слово, перенесённое через дефис, склеивается обратно."""
    if not lines:
        return ""
    base = min(ln.left for ln in lines)
    paragraphs, current = [], ""
    for i, ln in enumerate(lines):
        text = ln.text.strip()
        new_paragraph = i > 0 and (ln.left - base) > indent
        if new_paragraph and current:
            paragraphs.append(current)
            current = ""
        if current.endswith("-") and text[:1].islower() and not new_paragraph:
            current = current[:-1] + text              # «сбро-» + «шены» -> «сброшены»
        else:
            current = (current + " " + text).strip()
    if current:
        paragraphs.append(current)
    return "\n\n".join(re.sub(r"\s+", " ", p) for p in paragraphs)


def capital_start_ratio(lines):
    """Доля строк, начинающихся с заглавной буквы. В стихах так начинается почти
    каждая строка (≈0.9), в прозе строки обрываются посреди фразы (≈0.2–0.4)."""
    firsts = []
    for ln in lines:
        letter = next((c for c in ln.text if c.isalpha()), "")
        if letter:
            firsts.append(letter.isupper())
    return float(np.mean(firsts)) if firsts else 0.0


def cyrillic_ratio(lines):
    """Доля кириллицы среди непробельных знаков. На нотах, рисунках и мусоре низкая."""
    chars = [c for ln in lines for c in ln.text if not c.isspace()]
    if not chars:
        return 0.0
    cyr = sum(1 for c in chars if "а" <= c.lower() <= "я" or c.lower() == "ё")
    return cyr / len(chars)


def word_quality(lines):
    """Доля «настоящих» слов (3 буквы и больше) среди всех слов. У связного текста
    0.7–0.8, у нот и мусора OCR — 0.2–0.5: там россыпь одиночных знаков и слогов."""
    words = [w for ln in lines for w in re.findall(r"[^\W\d_]+", ln.text)]
    if not words:
        return 0.0
    return sum(1 for w in words if len(w) >= 3) / len(words)


@dataclass
class TextStructure:
    """Evidence about physical OCR lines, not sentences reconstructed by join_lines."""
    lines: int
    continuation: float
    hyphenated: float
    deep_indent: float
    right_flush: float | None
    stanza_breaks: float


def text_structure(lines, line_h):
    """Measure wrapping and layout without assuming that short sentences are verse.

    Margins are fitted independently to allow photographed-page perspective.
    Optional coordinates let custom
    engines keep using OcrLine(text, left); unavailable evidence stays unavailable.
    Repeated large indents differ from ordinary first-line paragraph indents.
    """
    lines = [ln for ln in lines if sum(c.isalpha() for c in ln.text) >= 3]
    n = len(lines)
    if not n:
        return TextStructure(0, 0.0, 0.0, 0.0, None, 0.0)
    line_h = max(1, line_h)
    left = np.array([ln.left for ln in lines], dtype=float)
    located = all(ln.right is not None and ln.top is not None and ln.bottom is not None
                  for ln in lines)
    right_flush, stanza_breaks = None, 0.0
    width = 0.0
    if located and n >= 3:
        right = np.array([ln.right for ln in lines], dtype=float)
        centers = np.array([(ln.top + ln.bottom) / 2 for ln in lines])
        spans = right - left
        width = float(np.percentile(spans, 90))
        longer = spans >= np.median(spans)
        # A ragged right edge can itself have a strong downward trend. Use
        # the body-text LEFT margin for skew, not the edge being evaluated.
        baseline = left <= np.median(left)
        slope = 0.0
        if baseline.sum() >= 3 and np.ptp(centers[baseline]) > 0:
            fitted = float(np.polyfit(centers[baseline], left[baseline], 1)[0])
            if abs(fitted) <= 0.25:  # A staircase is not a plausible text skew.
                slope = fitted
        right_slope = slope
        # Perspective can make the two margins converge. Fit the right edge
        # independently only with enough long lines; never apply that fit to
        # the left indents (a ragged right edge would invent a staircase).
        if longer.sum() >= 5 and np.ptp(centers[longer]) > 0:
            fitted = float(np.polyfit(centers[longer], right[longer], 1)[0])
            if abs(fitted) <= 0.25:
                right_slope = fitted
        right = right - right_slope * centers
        left = left - slope * centers
        margin = float(np.median(right[longer]))
        tolerance = max(0.6 * line_h, 0.025 * width)
        # The last line of a block can be an ordinary short paragraph ending.
        right_flush = float(np.mean(np.abs(right[:-1] - margin) <= tolerance))
        gaps = np.array([lines[i].top - lines[i - 1].bottom for i in range(1, n)])
        normal_gap = float(np.median(gaps))
        stanza_breaks = float(np.mean(gaps > normal_gap + 0.8 * line_h))
    base = float(np.percentile(left, 10))
    deep_indent = float(np.mean(left - base > max(3.0 * line_h, 0.12 * width)))
    continuation, hyphenated = 0, 0
    for previous, current in zip(lines, lines[1:]):
        first = next((c for c in current.text if c.isalpha()), "")
        end = previous.text.rstrip().rstrip('»”"\')]} ').rstrip()
        paragraph = current.left - previous.left > 0.9 * line_h
        if first.islower() and end and end[-1] not in ".!?…:;" and not paragraph:
            continuation += 1
        if re.search(r"[^\W\d_][-‐‑]\s*$", previous.text) and first.islower() and not paragraph:
            hyphenated += 1
    return TextStructure(n, continuation / max(1, n - 1), hyphenated / max(1, n - 1),
                         deep_indent, right_flush, stanza_breaks)
