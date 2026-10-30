"""
genre.py — дополнительный разбор уже распознанной страницы: какие блоки проза, какие стихи.

Работает поверх результата основного распознавания (координаты физических строк OCR),
страницу заново не читает. Главный признак, как в исследовании, — рваность правого края:
у прозы, набранной «в формат», строки доходят до края колонки, у стихов край рваный.
Решение о прозе принимает та же функция, что и у конвейера прозы (pipeline.prose_decision),
с теми же порогами ProseSettings.
"""
from dataclasses import dataclass
from statistics import median
from typing import List, Optional

from .blocks import ProseSettings
from .decorators import timed
from .ocr import OcrLine, capital_start_ratio, text_structure
from .pipeline import prose_decision

KINDS = {
    'prose': 'проза',
    'verse': 'стихи',
    'short': 'короткий фрагмент',
    'unknown': 'не определено',
}


@dataclass
class GenreVerdict:
    """Вывод по одному блоку: тип, рваность края и признаки, на которые он опирается."""
    index: int
    kind: str                       # prose / verse / short / unknown
    raggedness: Optional[float]     # доля строк, не дошедших до правого края (None — мало строк)
    lines: int
    capital_start: float            # доля строк, начинающихся с заглавной буквы
    reason: str

    @property
    def label(self) -> str:
        return KINDS[self.kind]

    def __str__(self):
        ragged = '—' if self.raggedness is None else f'{self.raggedness:.2f}'
        return f'Блок {self.index + 1}: {self.label} (рваность края {ragged}, строк {self.lines})'

    def to_dict(self) -> dict:
        return dict(index=self.index, kind=self.kind, label=self.label, lines=self.lines,
                    raggedness=None if self.raggedness is None else round(self.raggedness, 3),
                    capital_start=round(self.capital_start, 3), reason=self.reason)


class GenreAnalyzer:
    """Разбирает блоки на прозу и стихи. Пороги — композиция: свой экземпляр ProseSettings."""

    def __init__(self, settings: Optional[ProseSettings] = None):
        self.settings = settings or ProseSettings()

    @staticmethod
    def _lines(block: dict) -> List[OcrLine]:
        """Строки блока из ответа /api/ocr (line_boxes) -> OcrLine с координатами."""
        return [OcrLine(text=b['text'], left=b['x'], right=b['right'], top=b['y'], bottom=b['bottom'])
                for b in block.get('line_boxes', []) if b.get('text', '').strip()]

    def classify(self, index: int, block: dict) -> GenreVerdict:
        lines = self._lines(block)
        if not lines:
            return GenreVerdict(index, 'unknown', None, 0, 0.0, 'в блоке нет строк текста')
        line_h = max(1, round(median(ln.bottom - ln.top for ln in lines)))
        structure = text_structure(lines, line_h)
        raggedness = None if structure.right_flush is None else 1 - structure.right_flush
        capitals = capital_start_ratio(lines)
        s = self.settings
        if structure.lines == 0:              # только цифры и знаки: номер страницы, звездочки
            return GenreVerdict(index, 'unknown', None, 0, capitals, 'нет слов: номер страницы или знаки')
        if structure.lines < s.min_lines:
            return GenreVerdict(index, 'short', raggedness, structure.lines, capitals,
                                f'меньше {s.min_lines} строк: заголовок, подпись или короткая заметка')
        is_prose, reason = prose_decision(lines, line_h, s)
        if is_prose:
            kind, reason = 'prose', 'ровный правый край и признаки сплошного текста'
        elif reason == 'стихи':
            kind, reason = 'verse', ('почти все строки с заглавной буквы' if capitals > s.max_capital_start
                                     else 'строки «лесенкой»')
        elif reason == 'мало признаков прозы' and raggedness is not None and raggedness > 1 - s.min_ocr_flush:
            kind, reason = 'verse', 'рваный правый край'
        elif reason == 'не текст':
            kind, reason = 'unknown', 'похоже не на текст (ноты, рисунок, мусор OCR)'
        else:
            kind, reason = 'unknown', 'край ровный, но других признаков прозы мало'
        return GenreVerdict(index, kind, raggedness, structure.lines, capitals, reason)

    @timed
    def analyze(self, blocks: List[dict]) -> dict:
        verdicts = [self.classify(i, block) for i, block in enumerate(blocks)]
        counts = {kind: sum(v.kind == kind for v in verdicts) for kind in KINDS}
        return dict(blocks=[v.to_dict() for v in verdicts], counts=counts,
                    settings=dict(min_lines=self.settings.min_lines,
                                  max_raggedness=round(1 - self.settings.min_ocr_flush, 3),
                                  max_capital_start=self.settings.max_capital_start))
