"""Conservative, per-block deskew before OCR (not a prose/verse classifier)."""
import cv2
import numpy as np


def prepare_text(image):
    """Denoise textured paper only, then straighten; return image, angle, noise.

    Median filtering can erase thin strokes on clean scans. Use it only when
    the median pixel change is at least one gray level across the whole crop.
    This is a conservative noise proxy, not a scan/photo classifier.
    """
    median = cv2.medianBlur(image, 3)
    noise = float(np.median(cv2.absdiff(image, median)))
    cleaned = median if noise >= 1.0 else image
    corrected, angle = straighten_text(cleaned)
    return corrected, angle, noise


def straighten_text(image, max_angle=6.0):
    """Flatten text baselines by maximizing horizontal ink projection.

    Correct each column separately: the two halves of a photographed spread
    can lean in opposite directions. Never crop the rotated original. Return
    (image, correction_angle); ambiguous/already horizontal crops stay intact.
    """
    if min(image.shape[:2]) < 40:
        return image, 0.0
    scale = min(1.0, 600.0 / max(image.shape[:2]))
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, ink = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    # Ignore frames, border fragments, large pictures and speckle.
    allowed = np.zeros(n, dtype=np.uint8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if (area >= 3 and h < 0.20 * ink.shape[0]
                and w < 0.20 * ink.shape[1] and 0.1 < w / max(h, 1) < 12):
            allowed[i] = 1
    ink = allowed[labels].astype(np.float32)
    if ink.sum() < 100:
        return image, 0.0
    h, w = ink.shape
    center = (w / 2, h / 2)

    def score(angle):
        matrix = cv2.getRotationMatrix2D(center, float(angle), 1)
        rotated = cv2.warpAffine(ink, matrix, (w, h), flags=cv2.INTER_LINEAR)
        profile = rotated.sum(axis=1)
        return float(np.dot(profile, profile))

    angles = np.arange(-max_angle, max_angle + 0.125, 0.25)
    scores = [score(a) for a in angles]
    best = int(np.argmax(scores))
    angle = float(angles[best])
    baseline = score(0)
    if (abs(angle) < 0.5 or abs(angle) >= max_angle
            or scores[best] < 1.06 * baseline):
        return image, 0.0
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    nw = int(np.ceil(h * sine + w * cosine))
    nh = int(np.ceil(h * cosine + w * sine))
    matrix[0, 2] += (nw - w) / 2
    matrix[1, 2] += (nh - h) / 2
    return cv2.warpAffine(image, matrix, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255), angle
