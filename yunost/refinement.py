"""Conservative crop rereading: improve spelling without changing page geometry."""
from dataclasses import replace
from difflib import SequenceMatcher

import numpy as np

def refine_words(block, reread, confirm):
    """Only accept small, higher-confidence changes with one-to-one line geometry.

    Confidence is not ground truth. Keep the original spelling for inspection;
    never add/delete words, reflow verse, move a word, or merge page columns.
    """
    if len(reread) != 1:
        return 0
    candidate = reread[0]
    if candidate.confidence < block.confidence + 0.5:
        return 0
    if len(block.lines) != len(candidate.lines):
        return 0
    pairs = []
    for old_line, new_line in zip(block.lines, candidate.lines):
        if len(old_line) != len(new_line):
            return 0
        for old, new in zip(old_line, new_line):
            if (abs(old.x + old.w / 2 - new.x - new.w / 2) > max(4, 0.6 * old.h)
                    or abs(old.y + old.h / 2 - new.y - new.h / 2) > max(4, 0.6 * old.h)
                    or abs(old.w - new.w) > max(6, old.w * 0.25)):
                return 0
            pairs.append((old, new))
    changed = [(a, b) for a, b in pairs if a.text != b.text]
    if len(changed) > max(1, len(pairs) * 0.2):
        return 0
    replacements = {}
    for old, new in changed:
        if not any(c.isalpha() for c in old.text):
            continue
        # Do not silently replace commas, quotes, stress marks or line hyphens.
        if ''.join(c for c in old.text if not c.isalpha()) != ''.join(c for c in new.text if not c.isalpha()):
            continue
        if (new.confidence >= max(80, old.confidence + 2)
                and SequenceMatcher(None, old.text, new.text).ratio() >= 0.65
                and 0.8 <= len(new.text) / len(old.text) <= 1.25
                and confirm(old, new)):
            replacements[id(old)] = replace(old, text=new.text, confidence=new.confidence,
                                             original_text=old.text)
    block.words = [replacements.get(id(w), w) for w in block.words]
    return len(replacements)


def refine_blocks(blocks, flat, engine, max_blocks=12):
    remaining = 24

    def confirm(old, new):
        nonlocal remaining
        if remaining <= 0:
            return False
        remaining -= 1
        x, y, w, h = old.x, old.y, old.w, old.h
        words = [w for b in engine.read_layout(flat[max(0, y - 5):y + h + 5,
                                                   max(0, x - 4):x + w + 4], psm=7)
                 for w in b.words]
        # An isolated-word reading must agree on the letters. Its punctuation
        # can be unreliable at the crop boundary; the original marks stay intact.
        return (len(words) == 1 and words[0].confidence >= 80
                and ''.join(c for c in words[0].text if c.isalpha()) ==
                    ''.join(c for c in new.text if c.isalpha()))

    candidates = [b for b in blocks if not b.rejected and b.source == 'page'
                  and len(b.words) >= 4 and len(b.text) >= 20 and b.confidence < 95
                  and 6 <= np.median([w.h for w in b.words]) <= 30]
    for block in sorted(candidates, key=lambda b: b.confidence)[:max_blocks]:
        x, y, w, h = block.box
        x0, y0 = max(0, x - 8), max(0, y - 8)
        reread = engine.read_layout(flat[y0:y + h + 8, x0:x + w + 8], psm=6)
        for b in reread:
            for word in b.words:
                word.x += x0
                word.y += y0
        refine_words(block, reread, confirm)
