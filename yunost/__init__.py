"""yunost — распознавание текста на страницах газет и журналов и аналитика корпуса страниц."""
import logging
import os
import sys

# cvpipe.py лежит в корне проекта рядом с пакетом
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.getLogger("yunost").addHandler(logging.NullHandler())

from .blocks import BlockDetector, ProseSettings, TextBlock   # noqa: E402
from .ocr import OcrEngine, TesseractEngine                   # noqa: E402
from .pipeline import ProseExtractor, PageResult, UnreadableImage  # noqa: E402
from .text import TextExtractor, TextPageResult                  # noqa: E402

__all__ = ["BlockDetector", "ProseSettings", "TextBlock", "OcrEngine",
           "TesseractEngine", "TextExtractor", "TextPageResult",
           "ProseExtractor", "PageResult", "UnreadableImage"]
