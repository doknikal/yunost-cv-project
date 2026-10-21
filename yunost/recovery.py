"""Recover short paragraphs using a confirmed prose column, not a standalone genre guess."""
import numpy as np

from .blocks import TextBlock, _line_bands


def column_extensions(mask, blocks):
    """Propose expanded column crops around confirmed prose.

    Pixel components split at paragraph gaps and even inside a single wide line.
    Inspect the WHOLE width of a known column instead. Follow nearby physical
    lines of the same type size, stopping at whitespace, large headings and
    independently rejected long poems. No prose anchor means no recovery.
    """
    anchors = [b for b in blocks if b.is_prose]
    proposals = []
    for anchor in anchors:
        lh = anchor.line_h
        # Very wide introductions are not stable evidence for a body column.
        if anchor.w > 0.65 * mask.shape[1]:
            continue
        x0, x1 = anchor.x, anchor.x + anchor.w
        strip = mask[:, x0:x1]
        profile = np.count_nonzero(strip, axis=1)
        # Local threshold: a dense heading elsewhere must not erase faint text.
        threshold = max(3, 0.08 * np.percentile(profile[anchor.y:anchor.y + anchor.h], 90))
        bands = [band for band in _line_bands(np.where(profile >= threshold, 1.0, 0.0))
                 if band[-1] - band[0] + 1 >= 0.40 * lh]
        occupied = [i for i, band in enumerate(bands)
                    if band[-1] >= anchor.y and band[0] < anchor.y + anchor.h]
        if not occupied:
            continue
        first, last = min(occupied), max(occupied)
        original_first, original_last = first, last
        for direction in (-1, 1):
            current = first if direction < 0 else last
            while 0 <= current + direction < len(bands):
                index = current + direction
                band, previous = bands[index], bands[current]
                gap = (previous[0] - band[-1] if direction < 0 else band[0] - previous[-1])
                if gap > 2.0 * lh or len(band) > 2.0 * lh:
                    break
                y0, y1 = int(band[0]), int(band[-1]) + 1
                pixels = strip[y0:y1]
                counts = np.count_nonzero(pixels, axis=0)
                cumulative = np.cumsum(counts)
                # Ignore isolated speckle that would falsely connect a centred
                # author name to the left margin of the prose column.
                cols = np.flatnonzero((cumulative >= 0.03 * cumulative[-1]) &
                                      (cumulative <= 0.97 * cumulative[-1]))
                if len(cols) < 4 or y1 - y0 < 0.40 * lh:
                    break
                span = int(cols[-1] - cols[0] + 1)
                if np.count_nonzero(pixels) / max(1, (y1 - y0) * span) < 0.10:
                    break  # A tilted frame can project to a text-height band.
                if direction < 0 and span < 0.70 * anchor.w:
                    break  # Do not prepend an author/title to an introduction.
                # Exclude a neighbouring column/centred caption at a different margin.
                if cols[0] > max(3 * lh, 0.15 * anchor.w):
                    break
                protected = any(not b.is_prose and b.raw_text and
                                b.ocr_metrics.get('lines', 0) >= 8 and
                                b.rejected in ('стихи', 'мало признаков прозы') and
                                b.y <= (y0 + y1) / 2 < b.y + b.h and
                                min(x1, b.x + b.w) - max(x0, b.x) > 0.5 * anchor.w
                                for b in blocks)
                if protected:
                    break
                current = index
            if direction < 0:
                first = current
            else:
                last = current
        if first == original_first and last == original_last:
            continue
        y0 = min(anchor.y, int(bands[first][0]))
        y1 = max(anchor.y + anchor.h, int(bands[last][-1]) + 1)
        proposal = TextBlock(x0, y0, x1 - x0, y1 - y0, last - first + 1,
                             anchor.flush, lh, True)
        # Equivalent proposals from adjacent anchors are read only once.
        if any(abs(p.x - proposal.x) < lh and abs(p.w - proposal.w) < 2 * lh and
               p.overlap_with(proposal) >= 0.95 * min(p.area, proposal.area)
               for p in proposals):
            continue
        proposals.append(proposal)
    return proposals
