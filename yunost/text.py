"""All-text extraction. The legacy prose classifier is deliberately not called."""
from dataclasses import dataclass

import cv2
import numpy as np

import cvpipe
from .decorators import timed
from .layout import LayoutBlock, layout_order
from .ocr import TesseractEngine
from .pipeline import load_page


@dataclass
class TextPageResult:
    blocks: list
    gray: np.ndarray

    @property
    def text_blocks(self):
        return [b for b in self.blocks if not b.rejected]

    @property
    def text(self):
        return '\n\n'.join(b.text for b in self.text_blocks)

    def overlay_png(self):
        vis = cv2.cvtColor(self.gray, cv2.COLOR_GRAY2BGR)
        for number, block in enumerate(self.blocks, 1):
            x, y, w, h = block.box
            color = (150, 150, 150) if block.rejected else (60, 160, 0)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 3)
            cv2.putText(vis, str(number), (x, max(25, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        ok, png = cv2.imencode('.png', cv2.resize(vis, None, fx=0.5, fy=0.5,
                                                interpolation=cv2.INTER_AREA))
        return png.tobytes()


def _word_coverage(word, mask):
    region = mask[max(0, word.y):word.y + word.h, max(0, word.x):word.x + word.w]
    return float(region.mean()) if region.size else 0.0


def _mark_words(words, mask):
    for word in words:
        mask[max(0, word.y - 2):word.y + word.h + 2,
             max(0, word.x - 2):word.x + word.w + 2] = 1


def _text_evidence(block, sparse=False):
    """Only reject low-confidence, isolated OCR debris; no genre/length gate.

    One letter, digits, non-Cyrillic text and repeated short words are legitimate.
    Confidence is an OCR score, not a calibrated correctness probability.
    """
    text = ''.join(w.text for w in block.words)
    letters = [w for w in block.words if any(c.isalnum() for c in w.text)]
    real_size = any(w.h >= 6 and w.w <= 5 * w.h * len(w.text) for w in letters)
    threshold = 60 if sparse and len(text) <= 3 else 25
    return (bool(letters) and real_size and block.confidence >= threshold
            or any(c in '*⁎' for c in text) and block.confidence >= 35)


class TextExtractor:
    """Local page-layout OCR with a spatially deduplicated sparse-text fallback."""

    def __init__(self, engine=None):
        # Upscaling full pages can join staircase columns in Tesseract's layout
        # analysis. Keep the 2000px page scale; legacy crop OCR still uses 1.5x.
        self.engine = engine or TesseractEngine(scale=1.0)

    @timed
    def extract(self, data):
        gray = load_page(data)
        # Do not rotate a whole spread: columns can lean in opposite directions.
        flat = cvpipe.normalize_illumination(gray)
        blocks = self.engine.read_layout(flat, psm=3)
        mask = np.zeros(gray.shape, np.uint8)
        for block in blocks:
            if not _text_evidence(block):
                block.rejected = 'низкая уверенность OCR / нетекстовые знаки'
            else:
                _mark_words(block.words, mask)

        for sparse in self.engine.read_layout(flat, psm=11):
            missing = [w for w in sparse.words if _word_coverage(w, mask) < 0.35]
            if not missing or not _text_evidence(LayoutBlock(missing), sparse=True):
                continue
            candidate = LayoutBlock(missing, 'sparse')
            if not any(c.isalnum() for w in missing for c in w.text):
                _mark_words(candidate.words, mask)
                blocks.append(candidate)
                continue
            # Reread short isolated regions with enough left padding for large
            # initials. Sparse segmentation can omit the first glyph of a title.
            x, y, w, h = candidate.box
            pad_x, pad_y = min(120, max(8, round(1.5 * h))), 8
            x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
            crop = flat[y0:y + h + pad_y, x0:x + w + pad_x]
            reread = self.engine.read_layout(crop, psm=6)
            recovered = []
            for block in reread:
                for word in block.words:
                    word.x += x0
                    word.y += y0
                    if _word_coverage(word, mask) < 0.35:
                        recovered.append(word)
            if (recovered and _text_evidence(LayoutBlock(recovered), sparse=True)
                    and LayoutBlock(recovered).confidence >= 60):
                candidate = LayoutBlock(recovered, 'sparse')
            _mark_words(candidate.words, mask)
            blocks.append(candidate)
        accepted = layout_order([b for b in blocks if not b.rejected])
        rejected = [b for b in blocks if b.rejected]
        return TextPageResult(accepted + rejected, gray)
