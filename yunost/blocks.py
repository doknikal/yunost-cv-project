"""
blocks.py — поиск блоков текста на странице и отбор среди них ПРОЗЫ.

Идея. Проза в журнале набрана «в формат»: правые края строк почти совпадают.
У стихов правый край рваный. Поэтому для каждого блока считаем долю строк,
которые доходят до правого края блока («ровность»), и по ней отличаем прозу.
Заголовки, иллюстрации, рамки и стихи в OCR не попадают.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

import cvpipe


@dataclass
class ProseSettings:
    """Пороги отбора прозы. Меняются без правки кода."""
    min_lines: int = 8            # в блоке прозы не меньше стольких строк (короче — мало данных)
    flush_min: float = 0.65       # доля строк с ровным правым краем
    max_capital_start: float = 0.75   # больше строк с заглавной — это стихи (проверка после OCR)
    min_cyrillic: float = 0.75        # меньше кириллицы — ноты, рисунок или мусор
    min_word_quality: float = 0.60    # меньше настоящих слов — мусор OCR (ноты и т.п.)
    flush_tol: float = 0.015      # допуск «ровности» (доля ширины блока)
    min_fill: float = 0.04        # доля «чернил» в рамке блока (отсекает рамки и кромки фото)
    min_width: float = 0.10       # блок не уже этой доли ширины страницы
    contained: float = 0.85       # блок внутри другого блока на столько — дубль


@dataclass
class TextBlock:
    """Один блок текста на странице (координаты — в пикселях нормализованной страницы)."""
    x: int
    y: int
    w: int
    h: int
    n_lines: int
    flush: float                  # доля строк с ровным правым краем
    line_h: int                   # типичная высота строки на странице
    is_prose: bool = False
    rejected: str = field(default="", compare=False)   # почему не проза (после OCR)
    text: str = field(default="", compare=False)

    @property
    def box(self):
        return (self.x, self.y, self.w, self.h)

    @property
    def area(self):
        return self.w * self.h

    def overlap_with(self, other):
        """Площадь пересечения с другим блоком."""
        dx = min(self.x + self.w, other.x + other.w) - max(self.x, other.x)
        dy = min(self.y + self.h, other.y + other.h) - max(self.y, other.y)
        return max(0, dx) * max(0, dy)

    def __lt__(self, other):
        """Простое сравнение: выше — раньше, при равной высоте — левее."""
        return (self.y, self.x) < (other.y, other.x)

    def to_dict(self):
        return dict(x=self.x, y=self.y, w=self.w, h=self.h, lines=self.n_lines,
                    flush=round(self.flush, 2), is_prose=self.is_prose,
                    rejected=self.rejected, text=self.text)


def reading_order(blocks):
    """Порядок чтения: колонки слева направо, внутри колонки сверху вниз.
    Блок попадает в колонку, если перекрывается с ней по ширине больше чем наполовину."""
    columns = []                                  # каждая колонка: [x0, x1, [блоки]]
    for b in sorted(blocks, key=lambda b: b.x):
        for col in columns:
            shared = min(col[1], b.x + b.w) - max(col[0], b.x)
            if shared > 0.5 * min(col[1] - col[0], b.w):
                col[0], col[1] = min(col[0], b.x), max(col[1], b.x + b.w)
                col[2].append(b)
                break
        else:
            columns.append([b.x, b.x + b.w, [b]])
    columns.sort(key=lambda c: c[0])
    return [b for col in columns for b in sorted(col[2])]


def _close(img, width, height):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, height))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def _line_bands(profile):
    """Строки внутри блока: группы соседних строк пикселей с чернилами."""
    rows = np.where(profile > 0.20 * profile.max())[0]
    if len(rows) == 0:
        return []
    groups, current = [], [rows[0]]
    for r in rows[1:]:
        if r - current[-1] <= 3:
            current.append(r)
        else:
            groups.append(current)
            current = [r]
    groups.append(current)
    return groups


class BlockDetector:
    """Находит блоки текста и помечает, какие из них — проза."""

    def __init__(self, settings=None):
        self.settings = settings or ProseSettings()

    def detect(self, gray):
        """gray — страница в оттенках серого. Возвращает список TextBlock."""
        s = self.settings
        mask = cvpipe.ink_mask(gray)
        illo, _ = cvpipe.detect_illustrations(mask)
        text = cv2.bitwise_and(mask, cv2.bitwise_not(illo))   # иллюстрации — не текст
        text = self._strip_frames(text)
        text = self._strip_vertical_lines(text)
        text = self._strip_horizontal_lines(text)
        height, width = text.shape

        line_h = self._line_height(text)
        bars = _close(text, max(15, line_h), 1)                 # слова -> строки
        merged = _close(bars, 1, max(2, int(round(0.9 * line_h))))  # строки -> блок
        n, _, stats, _ = cv2.connectedComponentsWithStats(merged, 8)

        blocks = []
        for i in range(1, n):
            x, y, w, h, _ = (int(v) for v in stats[i])
            if w < s.min_width * width or h < 2 * line_h:
                continue
            region = text[y:y + h, x:x + w]
            if region.mean() / 255.0 < s.min_fill:              # рамка или кромка, а не текст
                continue
            block = self._measure(region, x, y, w, h, line_h)
            if block is not None:
                block.is_prose = (block.n_lines >= s.min_lines
                                  and block.flush >= s.flush_min)
                blocks.append(block)
        return self._drop_duplicates(blocks)

    @staticmethod
    def _strip_frames(text):
        """Убираем рамки и длинные линии: это протяжённые связные контуры почти без заливки
        (на фото рамка часто разорвана на куски, поэтому порог длины невысокий).
        Буквы между собой не связаны, поэтому текст такой фильтр не задевает."""
        height, width = text.shape
        n, labels, stats, _ = cv2.connectedComponentsWithStats(text, 8)
        cleaned = text.copy()
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if (w > 0.25 * width or h > 0.15 * height) and area < 0.25 * w * h:
                cleaned[labels == i] = 0
        return cleaned

    @staticmethod
    def _strip_vertical_lines(text):
        """Убираем длинные вертикальные штрихи (остатки рамок и края колонок):
        у букв сплошного вертикального штриха в 6% высоты страницы не бывает."""
        height = text.shape[0]
        wide = cv2.dilate(text, np.ones((1, 5), np.uint8))       # терпим небольшой наклон
        tall = cv2.getStructuringElement(cv2.MORPH_RECT, (3, int(0.06 * height)))
        lines = cv2.morphologyEx(wide, cv2.MORPH_OPEN, tall)
        lines = cv2.dilate(lines, np.ones((1, 7), np.uint8))
        return cv2.bitwise_and(text, cv2.bitwise_not(lines))

    @staticmethod
    def _strip_horizontal_lines(text):
        """Убираем длинные горизонтальные линии (нотный стан, линейки): строка текста
        не бывает сплошной на ширину в 12% страницы."""
        width = text.shape[1]
        wide = cv2.dilate(text, np.ones((5, 1), np.uint8))       # терпим небольшой наклон
        long_ = cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.12 * width), 1))
        lines = cv2.morphologyEx(wide, cv2.MORPH_OPEN, long_)
        lines = cv2.dilate(lines, np.ones((7, 1), np.uint8))
        return cv2.bitwise_and(text, cv2.bitwise_not(lines))

    @staticmethod
    def _line_height(text):
        """Типичная высота строки: склеиваем слова в полоски и берём медиану."""
        height, width = text.shape
        n, _, stats, _ = cv2.connectedComponentsWithStats(_close(text, 15, 1), 8)
        heights = [stats[i][3] for i in range(1, n)
                   if stats[i][2] > 0.08 * width and 3 < stats[i][3] < 0.05 * height]
        return int(np.median(heights)) if heights else 14

    def _measure(self, region, x, y, w, h, line_h):
        profile = region.sum(axis=1) / 255.0
        if profile.max() <= 0:
            return None
        edges, centers = [], []
        for band in _line_bands(profile):
            cols = np.where(region[band[0]:band[-1] + 1].sum(axis=0) > 0)[0]
            if len(cols):
                edges.append(cols.max())
                centers.append((band[0] + band[-1]) / 2.0)
        if len(edges) < 3:
            return None
        flush = self._flushness(np.array(edges[:-1]), np.array(centers[:-1]), w)
        return TextBlock(x, y, w, h, len(edges), flush, line_h)

    def _flushness(self, edges, centers, width):
        """Доля строк, правый край которых лежит на одной прямой.
        Прямую подгоняем по более длинным строкам (на фото страница может быть
        наклонена, поэтому край — не строго вертикаль); последняя строка абзаца
        в расчёт не идёт — она обычно короткая."""
        longer = edges >= np.median(edges)
        if longer.sum() < 2:
            return 0.0
        slope, shift = np.polyfit(centers[longer], edges[longer], 1)
        residual = np.abs(edges - (slope * centers + shift))
        return float(np.mean(residual <= max(4.0, self.settings.flush_tol * width)))

    def _drop_duplicates(self, blocks):
        """Если блок почти целиком лежит внутри другого, оставляем больший."""
        kept = []
        for b in sorted(blocks, key=lambda b: -b.area):
            inside = any(b.overlap_with(k) >= self.settings.contained * b.area
                         for k in kept)
            if not inside:
                kept.append(b)
        return reading_order(kept)
