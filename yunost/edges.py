"""Physical line-edge geometry, independent of text/genre recognition.

The historical projection-only metric remains available in cvpipe. This version
finds glyph-supported bands, retains short lines and ignores disconnected speckle.
It is a measurement inside ONE text column/block, not a page-layout classifier.
"""
import cv2
import numpy as np

from .preprocessing import straighten_text

METHOD = 'components-v2'


def _runs(active, gap):
    indices = np.flatnonzero(active)
    if not len(indices):
        return []
    breaks = np.flatnonzero(np.diff(indices) > gap + 1) + 1
    return [(int(part[0]), int(part[-1]) + 1) for part in np.split(indices, breaks)]


def _components(mask):
    return cv2.connectedComponentsWithStats(mask, 8)


def _solid_rule(w, h, area):
    """A solid elongated bar is not evidence of a typographic glyph scale."""
    return w >= 3 * h and area >= .85 * w * h


def _character_height(stats, shape):
    height, width = shape
    candidates = [h for x, y, w, h, area in stats[1:]
                  if 3 <= h <= max(6, height * .15) and area >= 4
                  and .08 <= w / h <= 5 and w < .5 * width
                  and not _solid_rule(w, h, area)]
    return float(np.median(candidates)) if candidates else None


def measure_edges(text_mask, col_span, y0, y1, *, deskew=True):
    """Return nullable std(right)/column_width plus auditable physical line boxes.

    Bounds are half-open original-image pixels. Out-of-image bounds are clipped;
    reversed/empty bounds produce an explicit empty result. No input is modified.
    Coordinates of the diagnostic boxes are mapped back after local deskew. The
    metric uses aligned coordinates, never a regression through right endpoints:
    genuinely stepped or progressively indented lines must remain irregular.
    """
    mask = np.asarray(text_mask)
    if mask.ndim != 2:
        raise ValueError('Ожидалась двумерная маска чернил')
    height, width = mask.shape
    x0, x1 = (max(0, min(width, int(x))) for x in col_span)
    y0, y1 = (max(0, min(height, int(y))) for y in (y0, y1))
    output = dict(method=METHOD, value=None, lines=[], line_count=0,
                  correction_angle=0.0, character_height=None,
                  bbox=[x0, y0, x1, y1], normalization='provided_region_width',
                  normalization_width=max(0, x1 - x0), geometry='local_rotation_only',
                  clipped_lines=0, central_spread=None, left_dispersion=None,
                  warnings=[])
    if x1 <= x0 or y1 <= y0:
        output['warnings'].append('empty_region')
        return output
    region = (mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
    if not np.any(region):
        output['warnings'].append('no_ink')
        return output
    matrix = np.array([[1., 0., 0.], [0., 1., 0.]])
    if deskew:
        _, angle = straighten_text(255 - region)
        if angle:
            h, w = region.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
            nw, nh = int(np.ceil(h * sine + w * cosine)), int(np.ceil(h * cosine + w * sine))
            matrix[0, 2] += (nw - w) / 2
            matrix[1, 2] += (nh - h) / 2
            region = cv2.warpAffine(region, matrix, (nw, nh), flags=cv2.INTER_NEAREST)
            output['correction_angle'] = angle
    inverse = cv2.invertAffineTransform(matrix)
    count, _, stats, _ = _components(region)
    glyph = _character_height(stats, region.shape)
    if glyph is None:
        output['warnings'].append('no_text_components')
        return output
    output['character_height'] = glyph
    core_ids, small_ids = [], []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if (.45 * glyph <= h <= 2.5 * glyph and w <= 12 * glyph
                and area >= max(3, .06 * glyph ** 2) and not _solid_rule(w, h, area)):
            core_ids.append(index)
        elif h <= 1.3 * glyph and w <= 2 * glyph and area >= max(1, .015 * glyph ** 2):
            small_ids.append(index)
    # Ink projections merge a descender with the next line's ascender. Cluster
    # character centres instead: physical lines need not have a completely blank
    # pixel row between them (especially staggered verse and compact type).
    center_rows = np.zeros(region.shape[0], dtype=np.uint8)
    centers = {i: stats[i, 1] + stats[i, 3] / 2 for i in core_ids}
    for center in centers.values():
        center_rows[min(len(center_rows) - 1, int(center))] = 1
    center_bands = _runs(center_rows, gap=max(1, round(.4 * glyph)))
    for start, end in center_bands:
        ids = [i for i in core_ids if start <= centers[i] < end]
        top = min(int(stats[i, 1]) for i in ids)
        bottom = max(int(stats[i, 1] + stats[i, 3]) for i in ids)
        left = min(int(stats[i, 0]) for i in ids)
        right = max(int(stats[i, 0] + stats[i, 2]) for i in ids)
        # Attach punctuation only when it is close to actual glyphs on this line.
        # A far-away pixel or dot must not pin every line to the column boundary.
        extras = [i for i in small_ids if stats[i, 1] < bottom + .2 * glyph
                  and stats[i, 1] + stats[i, 3] > top - .2 * glyph
                  and any(stats[i, 0] <= stats[j, 0] + stats[j, 2] + 1.5 * glyph
                          and stats[i, 0] + stats[i, 2] >= stats[j, 0] - 1.5 * glyph for j in ids)]
        all_ids = ids + extras
        left = min(int(stats[i, 0]) for i in all_ids)
        right = max(int(stats[i, 0] + stats[i, 2]) for i in all_ids)
        top = min(int(stats[i, 1]) for i in all_ids)
        bottom = max(int(stats[i, 1] + stats[i, 3]) for i in all_ids)
        corners = np.array([[left, top, 1], [right, top, 1],
                            [left, bottom, 1], [right, bottom, 1]], dtype=float) @ inverse.T
        bounds = [max(x0, int(np.floor(corners[:, 0].min())) + x0),
                  max(y0, int(np.floor(corners[:, 1].min())) + y0),
                  min(x1, int(np.ceil(corners[:, 0].max())) + x0),
                  min(y1, int(np.ceil(corners[:, 1].max())) + y0)]
        endpoint = inverse @ np.array([right, (top + bottom) / 2, 1.])
        clipped = (bounds[0] <= x0 or bounds[1] <= y0 or bounds[2] >= x1 or bounds[3] >= y1)
        output['clipped_lines'] += int(clipped)
        output['lines'].append(dict(box=bounds, right=float(endpoint[0] + x0),
                                    center_y=float(endpoint[1] + y0), aligned_right=right,
                                    aligned_left=left, clipped=clipped))
    output['line_count'] = len(output['lines'])
    if len(output['lines']) >= 4:
        edges = [line['aligned_right'] for line in output['lines']]
        output['value'] = float(np.std(edges) / (x1 - x0))
        output['central_spread'] = float(np.diff(np.percentile(edges, [10, 90]))[0] / (x1 - x0))
        output['left_dispersion'] = float(np.std([line['aligned_left'] for line in output['lines']]) / (x1 - x0))
    else:
        output['warnings'].append('fewer_than_four_lines')
    if output['clipped_lines']:
        output['warnings'].append('ink_touches_region_boundary')
    return output


def analyze_page_edges(result):
    """Measure accepted OCR blocks without changing OCR text or its acceptance.

    OCR provides the local region only; character and endpoint geometry comes
    from pixels, not word confidence or recognized spelling. The normalization
    is occupied OCR-block width plus padding, NOT the historical page-column
    metric. Uploaded pages/results are not persisted.
    """
    import cvpipe
    output = dict(method=METHOD, normalization='padded_ocr_block_width',
                  blocks=[], value=None, warnings=[])
    if not len(result):
        return output
    mask = cvpipe.ink_mask(result.gray)
    for number, block in enumerate(result.blocks, 1):
        if block.rejected:
            continue
        x, y, w, h = block.box
        measurement = measure_edges(mask, (x - 4, x + w + 4), y - 4, y + h + 4)
        measurement['block_number'] = number
        output['blocks'].append(measurement)
    usable = [b for b in output['blocks'] if b['value'] is not None]
    if usable:
        output['value'] = float(np.sqrt(sum(b['line_count'] * b['value'] ** 2 for b in usable)
                                       / sum(b['line_count'] for b in usable)))
    else:
        output['warnings'].append('no_block_with_four_supported_lines')
    return output
