"""
ocr.py — движки распознавания текста.

Движок получает картинку ОДНОГО блока текста и возвращает строки с отступами.
Чтобы подключить другой OCR (например, PaddleOCR), достаточно написать
наследника OcrEngine с методом read_lines().
"""
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class OcrLine:
    text: str
    left: int          # отступ слева в пикселях (нужен, чтобы найти начало абзаца)


class OcrEngine(ABC):
    """Общий интерфейс: блок текста (numpy-картинка) -> текст."""

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
    """Tesseract (бесплатный, работает на обычном процессоре)."""

    name = "tesseract"

    def __init__(self, lang="rus", scale=1.5):
        import pytesseract                         # импорт здесь, чтобы тесты без OCR работали
        self._tess = pytesseract
        self.lang = lang
        self.scale = scale

    def read_lines(self, image):
        import cv2
        if self.scale != 1:
            image = cv2.resize(image, None, fx=self.scale, fy=self.scale,
                               interpolation=cv2.INTER_CUBIC)
        data = self._tess.image_to_data(image, lang=self.lang, config="--psm 6",
                                        output_type=self._tess.Output.DICT)
        lines = {}
        for i, word in enumerate(data["text"]):
            if not word.strip() or float(data["conf"][i]) < 0:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            entry = lines.setdefault(key, {"words": [], "left": data["left"][i]})
            entry["words"].append(word)
            entry["left"] = min(entry["left"], data["left"][i])
        ordered = sorted(lines.items(), key=lambda kv: kv[0])
        return [OcrLine(" ".join(v["words"]), int(v["left"] / self.scale))
                for _, v in ordered]


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
