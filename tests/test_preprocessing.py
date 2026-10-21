import cv2
import numpy as np
import pytest

from yunost.preprocessing import prepare_text, straighten_text


def text_sample():
    image = np.full((350, 650), 255, dtype=np.uint8)
    for y in range(40, 330, 30):
        cv2.putText(image, "Several words in a printed line.", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 1, cv2.LINE_AA)
    return image


def test_horizontal_and_blank_images_are_not_rotated():
    for image in [text_sample(), np.full((300, 400), 255, np.uint8)]:
        result, angle = straighten_text(image)
        assert angle == 0 and np.array_equal(result, image)


def test_preparation_preserves_clean_scan():
    image = text_sample()
    result, angle, noise = prepare_text(image)
    assert noise == 0 and angle == 0
    assert np.array_equal(result, image)


def test_preparation_denoises_textured_background():
    rng = np.random.default_rng(42)
    image = np.clip(text_sample().astype(float) - 15 + rng.normal(0, 5, (350, 650)),
                    0, 255).astype(np.uint8)
    result, angle, noise = prepare_text(image)
    assert noise >= 1 and angle == 0
    assert result[:15].std() < image[:15].std()


@pytest.mark.parametrize("angle", [-3.0, 3.0])
def test_tilt_is_corrected_without_clipping(angle):
    image = text_sample()
    matrix = cv2.getRotationMatrix2D((325, 175), angle, 1)
    tilted = cv2.warpAffine(image, matrix, (650, 350), borderValue=255)
    result, correction = straighten_text(tilted)
    assert abs(correction + angle) <= 0.5
    assert result.shape[0] >= image.shape[0]
    assert result.shape[1] >= image.shape[1]


def test_detector_recovers_lines_in_tilted_wide_paragraph():
    from yunost.blocks import BlockDetector
    image = text_sample()
    matrix = cv2.getRotationMatrix2D((325, 175), 3.0, 1)
    tilted = cv2.warpAffine(image, matrix, (650, 350), borderValue=255)
    _, ink = cv2.threshold(tilted, 128, 255, cv2.THRESH_BINARY_INV)
    block = BlockDetector()._measure(ink, 0, 0, 650, 350, 17)
    assert block is not None and block.n_lines >= 8
