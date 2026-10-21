"""Genre-neutral OCR geometry and physical-line output (normalized page pixels)."""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class LayoutWord:
    text: str
    x: int
    y: int
    w: int
    h: int
    confidence: float
    line_id: tuple


@dataclass
class LayoutBlock:
    words: list
    source: str = 'page'
    rejected: str = ''

    @property
    def box(self):
        x = min(w.x for w in self.words)
        y = min(w.y for w in self.words)
        return (x, y, max(w.x + w.w for w in self.words) - x,
                max(w.y + w.h for w in self.words) - y)

    @property
    def lines(self):
        groups = {}
        for word in self.words:
            groups.setdefault(word.line_id, []).append(word)
        # Tesseract's line IDs preserve physical reading order within a block.
        return [sorted(words, key=lambda w: w.x) for words in groups.values()]

    @property
    def confidence(self):
        return sum(w.confidence * len(w.text) for w in self.words) / max(
            1, sum(len(w.text) for w in self.words))

    @property
    def text(self):
        """Never join physical lines or silently remove typographical hyphens."""
        lines = self.lines
        base = self.box[0]
        char_width = float(np.median([w.w / max(1, len(w.text)) for w in self.words]))
        heights = [max(w.h for w in line) for line in lines]
        typical_height = float(np.median(heights))
        output = []
        previous_bottom = None
        for line in lines:
            top = min(w.y for w in line)
            if previous_bottom is not None and top - previous_bottom > 1.3 * typical_height:
                output.append('')
            indent = min(80, max(0, round((line[0].x - base) / max(1, char_width))))
            output.append(' ' * indent + ' '.join(w.text for w in line))
            previous_bottom = max(w.y + w.h for w in line)
        return '\n'.join(output)

    def to_dict(self):
        x, y, w, h = self.box
        return dict(x=x, y=y, w=w, h=h, lines=len(self.lines), text=self.text,
                    confidence=round(self.confidence, 1), source=self.source,
                    is_text=not bool(self.rejected), rejected=self.rejected,
                    line_boxes=[dict(x=min(w.x for w in line), y=min(w.y for w in line),
                                     right=max(w.x + w.w for w in line),
                                     bottom=max(w.y + w.h for w in line),
                                     text=' '.join(w.text for w in line)) for line in self.lines],
                    words=[asdict(w) for w in self.words])


def layout_order(blocks):
    """Recursive whitespace cuts: sections top-to-bottom, columns left-to-right.

    A full-width heading/footer prevents a vertical cut until its horizontal
    section has been separated. Unlike overlap clustering, it cannot merge
    unrelated body columns into a single top-to-bottom stream.
    """
    if len(blocks) < 2:
        return list(blocks)

    def gaps(axis):
        spans = sorted((b.box[axis], b.box[axis] + b.box[axis + 2]) for b in blocks)
        end = spans[0][1]
        found = []
        for start, stop in spans[1:]:
            if start > end:
                found.append((start - end, (start + end) / 2))
            end = max(end, stop)
        return found

    vertical, horizontal = gaps(0), gaps(1)
    typical_height = float(np.median([w.h for b in blocks for w in b.words]))
    if horizontal and max(horizontal)[0] >= 2 * typical_height:
        axis, (_, cut) = 1, max(horizontal)
    elif vertical:
        axis, (_, cut) = 0, max(vertical)
    elif horizontal:
        axis, (_, cut) = 1, max(horizontal)
    else:
        return sorted(blocks, key=lambda b: (b.box[1], b.box[0]))
    before = [b for b in blocks if b.box[axis] < cut]
    after = [b for b in blocks if b.box[axis] >= cut]
    return layout_order(before) + layout_order(after)
