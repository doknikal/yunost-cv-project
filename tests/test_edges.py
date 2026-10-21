"""Synthetic physical-geometry contracts; no scans, OCR or annotations are used."""
import cv2
import numpy as np
import pytest

from yunost.edges import measure_edges


DEFAULT_SPECS = [
    (20, 24, 28),
    (132, 56, 1),
    (44, 88, 10),
    (184, 120, 2),
    (76, 152, 18),
    (212, 184, 4),
]


def paragraph(specs=None, shape=(240, 400), height=12, glyph_width=6, pitch=10):
    """Draw separated rectangular glyphs; specs are (left, top, glyph_count).

    Rectangles deliberately make half-open ink endpoints exact. In particular,
    a one-glyph line is real supported text, not an image-wide projection peak.
    """
    mask = np.zeros(shape, np.uint8)
    boxes = []
    for left, top, count in DEFAULT_SPECS if specs is None else specs:
        right = left + (count - 1) * pitch + glyph_width
        assert count > 0 and 0 <= left < right <= shape[1]
        assert 0 <= top < top + height <= shape[0]
        for index in range(count):
            x = left + index * pitch
            mask[top:top + height, x:x + glyph_width] = 255
        boxes.append([left, top, right, top + height])
    return mask, boxes


def assert_exact_lines(result, boxes, crop_left=0):
    assert result['method'] == 'components-v2'
    assert result['line_count'] == len(boxes) == len(result['lines'])
    for line, box in zip(sorted(result['lines'], key=lambda item: item['center_y']), boxes):
        assert line['box'] == box
        assert line['right'] == pytest.approx(box[2])
        assert line['center_y'] == pytest.approx((box[1] + box[3]) / 2)
        assert line['aligned_right'] == pytest.approx(box[2] - crop_left)


def measure_full(mask, **kwargs):
    return measure_edges(mask, (0, mask.shape[1]), 0, mask.shape[0], **kwargs)


def test_all_short_and_indented_lines_are_retained():
    mask, boxes = paragraph()
    result = measure_full(mask, deskew=False)
    assert_exact_lines(result, boxes)
    # Includes one-, two- and four-glyph lines beside a 28-glyph line. Their
    # low row projection and indentation must not make them disappear.
    assert result['value'] == pytest.approx(np.std([box[2] for box in boxes]) / 400)
    assert result['character_height'] == pytest.approx(12)


def test_known_population_std_uses_original_crop_width():
    specs = [(24, 30 + 32 * i, count) for i, count in enumerate([6, 10, 14, 18])]
    mask, boxes = paragraph(specs)
    assert [box[2] for box in boxes] == [80, 120, 160, 200]
    result = measure_edges(mask, (14, 354), 10, 220, deskew=False)
    assert_exact_lines(result, boxes, crop_left=14)
    # Mean = 140, population variance = (3600 + 400 + 400 + 3600) / 4.
    assert result['value'] == pytest.approx(np.sqrt(2000) / 340, abs=1e-12)


def test_aligned_right_edges_have_zero_std_even_with_different_left_edges():
    specs = [(24 + 20 * i, 24 + 32 * i, 28 - 2 * i) for i in range(6)]
    mask, boxes = paragraph(specs)
    assert len({box[2] for box in boxes}) == 1
    result = measure_full(mask, deskew=False)
    assert_exact_lines(result, boxes)
    assert result['value'] == pytest.approx(0, abs=1e-12)


def test_real_progressive_indentation_is_not_regressed_out_as_skew():
    specs = [(24 + 36 * i, 24 + 32 * i, 12) for i in range(6)]
    mask, boxes = paragraph(specs)
    result = measure_full(mask, deskew=True)
    assert_exact_lines(result, boxes)
    assert result['correction_angle'] == pytest.approx(0, abs=0.25)
    assert result['value'] == pytest.approx(np.std([box[2] for box in boxes]) / 400)
    assert result['value'] > 0.1


@pytest.mark.parametrize('every_ink_row', [False, True], ids=['per-line', 'per-scan-row'])
def test_distant_single_pixel_speckles_do_not_pin_endpoints(every_ink_row):
    mask, boxes = paragraph()
    for _, top, _, bottom in boxes:
        rows = range(top, bottom) if every_ink_row else [bottom - 2]
        for y in rows:
            # Alternation prevents adjacent scan rows becoming a vertical
            # connected component: each speckle really is one isolated pixel.
            mask[y, 370 if y % 2 else 390] = 255
    result = measure_full(mask, deskew=False)
    assert_exact_lines(result, boxes)
    assert result['value'] == pytest.approx(np.std([box[2] for box in boxes]) / 400)
    assert result['value'] > 0.1


@pytest.mark.parametrize('punctuation', ['period', 'comma'])
def test_supported_punctuation_is_kept_but_identical_far_noise_is_dropped(punctuation):
    mask, boxes = paragraph()
    expected = []
    for left, top, right, bottom in boxes:
        if punctuation == 'period':
            dot = np.full((2, 2), 255, np.uint8)
        else:
            dot = np.array([[255, 255], [255, 255], [255, 0]], np.uint8)
        y = bottom - dot.shape[0]
        mask[y:bottom, right + 4:right + 6] = dot
        mask[y:bottom, 384:386] = dot
        expected.append([left, top, right + 6, bottom])
    result = measure_full(mask, deskew=False)
    assert_exact_lines(result, expected)
    assert result['value'] == pytest.approx(np.std([box[2] for box in expected]) / 400)


@pytest.mark.parametrize('scale', [1, 2, 3])
def test_low_resolution_supported_periods_survive_resolution_changes(scale):
    # A 6-pixel-high glyph has a legitimate one-pixel full stop two pixels
    # after it. Proximity to text, not a fixed minimum dot area, is support.
    specs = [(left // 2, top // 2, count) for left, top, count in DEFAULT_SPECS]
    mask, boxes = paragraph(specs, (120, 200), height=6, glyph_width=3, pitch=5)
    expected = []
    for left, top, right, bottom in boxes:
        mask[bottom - 1, right + 2] = 255
        mask[bottom - 1, 192] = 255
        expected.append([left, top, right + 3, bottom])
    scaled = np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)
    result = measure_full(scaled, deskew=False)
    assert_exact_lines(result, [[coordinate * scale for coordinate in box] for box in expected])
    assert result['value'] == pytest.approx(np.std([box[2] for box in expected]) / 200)


@pytest.mark.parametrize('shape', [(240, 400), (1, 1), (2, 2)])
def test_empty_or_tiny_regions_have_null_measurement(shape):
    for fill in ([0, 255] if max(shape) <= 2 else [0]):
        result = measure_full(np.full(shape, fill, np.uint8), deskew=False)
        assert result['value'] is None
        assert result['line_count'] == 0
        assert result['lines'] == []
        assert result['warnings']


@pytest.mark.parametrize('count', [1, 2, 3])
def test_fewer_than_four_lines_are_reported_but_not_measured(count):
    mask, boxes = paragraph(DEFAULT_SPECS[:count])
    result = measure_full(mask, deskew=False)
    assert_exact_lines(result, boxes)
    assert result['value'] is None
    assert result['warnings']


@pytest.mark.parametrize('span,y0,y1', [
    ((50, 50), 0, 240),
    ((300, 20), 0, 240),
    ((-40, -1), 0, 240),
    ((450, 500), 0, 240),
    ((0, 400), 100, 100),
    ((0, 400), 200, 20),
    ((0, 400), -30, -1),
    ((0, 400), 260, 300),
])
def test_empty_reversed_or_fully_outside_bounds_return_null(span, y0, y1):
    mask, _ = paragraph()
    result = measure_edges(mask, span, y0, y1, deskew=False)
    assert result['value'] is None
    assert result['line_count'] == 0
    assert result['lines'] == []
    assert result['warnings']


def test_outside_bounds_are_clipped_before_normalizing_or_slicing():
    mask, boxes = paragraph()
    result = measure_edges(mask, (-40, 470), -30, 300, deskew=False)
    assert_exact_lines(result, boxes)
    assert result['value'] == pytest.approx(np.std([box[2] for box in boxes]) / 400)


def test_partial_crop_reports_clipped_ink_in_original_coordinates():
    mask, boxes = paragraph()
    x0, y0, x1, y1 = 23, 28, 254, 191
    expected = [[max(left, x0), max(top, y0), min(right, x1), min(bottom, y1)]
                for left, top, right, bottom in boxes]
    result = measure_edges(mask, (x0, x1), y0, y1, deskew=False)
    assert_exact_lines(result, expected, crop_left=x0)
    assert result['value'] == pytest.approx(np.std([box[2] for box in expected]) / (x1 - x0))


@pytest.mark.parametrize('move_crop', [False, True], ids=['ink-only', 'ink-and-crop'])
def test_std_is_translation_invariant(move_crop):
    mask, boxes = paragraph()
    dx, dy = 37, 17
    if move_crop:
        shifted = np.zeros((mask.shape[0] + dy + 20, mask.shape[1] + dx + 20), np.uint8)
        shifted[dy:dy + mask.shape[0], dx:dx + mask.shape[1]] = mask
        span, y0, y1 = (dx, dx + mask.shape[1]), dy, dy + mask.shape[0]
    else:
        shifted = np.zeros_like(mask)
        shifted[dy:, dx:] = mask[:-dy, :-dx]
        span, y0, y1 = (0, mask.shape[1]), 0, mask.shape[0]
    result = measure_edges(shifted, span, y0, y1, deskew=False)
    expected = [[left + dx, top + dy, right + dx, bottom + dy]
                for left, top, right, bottom in boxes]
    assert_exact_lines(result, expected, crop_left=span[0])
    assert result['value'] == pytest.approx(measure_full(mask, deskew=False)['value'], abs=1e-12)


@pytest.mark.parametrize('scale', [1, 2, 3, 4])
def test_normalized_std_and_line_geometry_are_resolution_invariant(scale):
    specs = [(left // 2, top // 2, count) for left, top, count in DEFAULT_SPECS]
    mask, boxes = paragraph(specs, (120, 200), height=6, glyph_width=3, pitch=5)
    scaled = np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)
    result = measure_full(scaled, deskew=False)
    assert_exact_lines(result, [[coordinate * scale for coordinate in box] for box in boxes])
    assert result['value'] == pytest.approx(np.std([box[2] for box in boxes]) / 200, abs=1e-12)
    assert result['character_height'] == pytest.approx(6 * scale)


def tilted_paragraph(angle):
    """Rotate each known line independently to retain exact raster ground truth.

    The generating canvas has ample margins. The measured crop, however, is
    tight to the rotated ink: a deskewer must expand it rather than lose tips.
    """
    shape = (640, 1000)
    specs = [(left, 100 + 36 * i, count) for i, (left, count) in enumerate([
        (96, 52), (122, 38), (96, 48), (148, 18), (96, 42),
        (174, 30), (96, 50), (200, 10), (96, 46), (226, 22),
    ])]
    matrix = cv2.getRotationMatrix2D((500, 320), angle, 1)
    mask = np.zeros(shape, np.uint8)
    rotated_boxes, endpoints, rights = [], [], []
    for spec in specs:
        line_mask, (box,) = paragraph([spec], shape, height=16, glyph_width=8, pitch=12)
        rotated = cv2.warpAffine(line_mask, matrix, (shape[1], shape[0]),
                                 flags=cv2.INTER_NEAREST, borderValue=0)
        mask |= rotated
        ys, xs = np.nonzero(rotated)
        rotated_boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
        endpoints.append(matrix @ np.array([box[2], (box[1] + box[3]) / 2, 1]))
        rights.append(box[2])
    crop = (min(box[0] for box in rotated_boxes), min(box[1] for box in rotated_boxes),
            max(box[2] for box in rotated_boxes), max(box[3] for box in rotated_boxes))
    return mask, rotated_boxes, endpoints, crop, np.std(rights)


@pytest.mark.parametrize('angle', [-3.0, 3.0])
def test_local_deskew_maps_boxes_and_endpoints_back_without_clipping(angle):
    mask, expected_boxes, expected_endpoints, (x0, y0, x1, y1), expected_std = tilted_paragraph(angle)
    assert x0 > 0 and y0 > 0  # A real local crop, not an origin-only transform.
    result = measure_edges(mask, (x0, x1), y0, y1, deskew=True)
    assert result['correction_angle'] == pytest.approx(-angle, abs=0.5)
    assert result['line_count'] == len(expected_boxes) == len(result['lines'])
    # Two nearest-neighbor rotations can move endpoints by roughly a pixel;
    # allow two pixels, not a percentage tolerance that hides lost short lines.
    assert result['value'] == pytest.approx(expected_std / (x1 - x0), abs=2 / (x1 - x0))
    assert result['value'] == pytest.approx(
        np.std([line['aligned_right'] for line in result['lines']]) / (x1 - x0), abs=1e-12)
    coverage = np.zeros_like(mask, dtype=bool)
    lines = sorted(result['lines'], key=lambda item: item['center_y'])
    for line, box, endpoint in zip(lines, expected_boxes, expected_endpoints):
        # Axis-aligned envelopes and rasterized rotated rectangles may differ
        # by a few pixels, but coordinates must be in the ORIGINAL page frame.
        np.testing.assert_allclose(line['box'], box, atol=3, rtol=0)
        np.testing.assert_allclose([line['right'], line['center_y']], endpoint, atol=2, rtol=0)
        left, top, right, bottom = line['box']
        assert x0 <= left < right <= x1
        assert y0 <= top < bottom <= y1
        coverage[max(0, top - 1):bottom + 1, max(0, left - 1):right + 1] = True
    assert not np.any((mask > 0) & ~coverage), 'some original line ink was clipped out of mapped boxes'


@pytest.mark.parametrize('deskew', [False, True])
@pytest.mark.parametrize('dtype', [np.uint8, np.bool_], ids=['uint8', 'bool'])
def test_input_mask_is_not_mutated_even_when_read_only(deskew, dtype):
    mask = tilted_paragraph(3)[0] if deskew else paragraph()[0]
    mask = mask.astype(dtype)
    before = mask.copy()
    mask.flags.writeable = False
    result = measure_full(mask, deskew=deskew)
    assert result['line_count'] >= 4
    np.testing.assert_array_equal(mask, before)


@pytest.mark.parametrize('kind', ['solid-picture', 'long-rules', 'short-thick-rules'])
def test_nontext_solid_picture_or_rules_are_not_a_series_of_text_lines(kind):
    mask = np.zeros((240, 400), np.uint8)
    if kind == 'solid-picture':
        mask[20:220, 40:360] = 255
    elif kind == 'long-rules':
        for i in range(6):
            mask[24 + 32 * i:27 + 32 * i, 40:340 - 10 * i] = 255
    else:
        # Six uniform solid 5:1 bars, not six groups of character strokes.
        # Unlike a short glyph-supported text line, no surrounding text gives
        # these horizontal rules evidence of a typographic character scale.
        for i in range(6):
            mask[24 + 32 * i:36 + 32 * i, 40:100] = 255
    result = measure_full(mask, deskew=False)
    assert result['value'] is None
    assert result['line_count'] < 4


def test_historical_projection_metric_is_preserved_independently_of_v2():
    # Exercise cvpipe directly, not the API or a future wrapper contract. The
    # historical 20%-of-maximum row cutoff intentionally drops the final short
    # line, whereas components-v2 must retain it.
    from cvpipe import right_edge_raggedness

    specs = [(24, 24 + 32 * i, count) for i, count in enumerate([25, 21, 17, 13, 9, 1])]
    mask, boxes = paragraph(specs)
    assert [box[2] for box in boxes[:5]] == [270, 230, 190, 150, 110]
    assert right_edge_raggedness(mask, (0, 400), 0, 240) == pytest.approx(np.sqrt(3200) / 400)
    result = measure_full(mask, deskew=False)
    assert_exact_lines(result, boxes)
    assert result['value'] == pytest.approx(np.std([box[2] for box in boxes]) / 400)


def test_rules_do_not_change_supported_text_edges():
    mask, boxes = paragraph()
    mask[:, 360:363] = 255
    mask[8:11, 8:390] = 255
    for top in [43, 75, 107, 139, 171, 203]:
        mask[top:top + 12, 285:345] = 255
    assert_exact_lines(measure_full(mask, deskew=False), boxes)


def test_public_method_dispatch_preserves_legacy_default():
    from cvpipe import analyze_right_edge, right_edge_raggedness
    mask, _ = paragraph()
    np.testing.assert_equal(right_edge_raggedness(mask, (0, 400), 0, 240),
                            right_edge_raggedness(mask, (0, 400), 0, 240, method='legacy'))
    assert right_edge_raggedness(mask, (0, 400), 0, 240, method='components-v2') == (
        analyze_right_edge(mask, (0, 400), 0, 240)['value'])
    assert np.isnan(right_edge_raggedness(np.zeros_like(mask), (0, 400), 0, 240,
                                        method='components-v2'))
    with pytest.raises(ValueError):
        right_edge_raggedness(mask, (0, 400), 0, 240, method='unknown')
