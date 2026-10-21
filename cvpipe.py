"""
cvpipe.py — функции компьютерного зрения для анализа страниц журнала «Юность».

Идея простая и без нейросетей (только OpenCV и NumPy):
  1) переводим страницу в чёрно-белую «маску чернил» (где текст/картинки);
  2) находим рамку текста, отделяем крупные иллюстрации;
  3) по форме текста считаем простые признаки вёрстки: плотность, число
      колонок, вариативность правого края и т.п. Геометрия не определяет жанр.

Каждая функция делает один маленький шаг. extract_features() собирает их вместе
и возвращает словарь признаков по одной странице.
"""
import cv2
import numpy as np
from scipy.signal import find_peaks


def _runs(flags):
    """Вспомогательная функция: находит непрерывные участки True в массиве.
    Например, [F,T,T,F,T] -> [(1,3),(4,5)]. Используется, чтобы находить
    колонки и строки текста."""
    runs = []
    i = 0
    while i < len(flags):
        if flags[i]:
            start = i
            while i < len(flags) and flags[i]:
                i += 1
            runs.append((start, i))   # участок [start, i)
        else:
            i += 1
    return runs


def load_gray(path):
    """Читаем картинку и сразу переводим в оттенки серого."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def normalize_illumination(gray):
    """Выравниваем освещение (важно для фото со смартфона, где есть тень).
    Оцениваем светлый фон и делим картинку на него — тёмный текст становится
    одинаково контрастным по всей странице."""
    k = max(31, gray.shape[0] // 30)        # размер ядра ~3% высоты страницы
    if k % 2 == 0:
        k += 1                              # OpenCV требует нечётный размер
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)   # оценка фона
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3.0)
    norm = (gray.astype(np.float32) + 1) / (bg.astype(np.float32) + 1)
    return np.clip(norm * 255, 0, 255).astype(np.uint8)


def ink_mask(gray):
    """Чёрно-белая маска: 255 там, где чернила (текст/картинки), иначе 0.
    Порог подбирается автоматически методом Оцу."""
    norm = normalize_illumination(gray)
    _, mask = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # убираем одиночные точки шума
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    return mask


def content_bbox(mask, frac=0.005):
    """Рамка содержимого (x0, y0, x1, y1): отрезаем пустые поля по краям.
    Берём строки и столбцы, где чернил заметно больше нуля."""
    H, W = mask.shape
    col_ink = mask.sum(axis=0) / 255.0      # сколько чернил в каждом столбце
    row_ink = mask.sum(axis=1) / 255.0      # сколько чернил в каждой строке
    cols = np.where(col_ink > frac * col_ink.max())[0]
    rows = np.where(row_ink > frac * row_ink.max())[0]
    if len(cols) == 0 or len(rows) == 0:
        return 0, 0, W, H
    return int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1


def detect_illustrations(mask, min_area_frac=0.012):
    """Находим крупные иллюстрации/фото. Сначала «склеиваем» близкие чернила
    в сплошные пятна, затем оставляем только большие и плотные — текст так не
    выглядит (он «дырявый»). Возвращаем маску иллюстраций и список рамок."""
    H, W = mask.shape
    closed = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
    illo = np.zeros_like(mask)
    boxes = []
    page_area = H * W
    for i in range(1, n):                   # 0 — это фон, пропускаем
        x, y, w, h, area = stats[i]
        if area < min_area_frac * page_area:
            continue                        # слишком маленькое — не иллюстрация
        solidity = area / float(w * h)      # насколько пятно заполняет рамку
        fill = mask[y:y + h, x:x + w].mean() / 255.0
        if solidity > 0.45 and fill > 0.35 and min(w, h) > 0.06 * H:
            illo[labels == i] = 255
            boxes.append((int(x), int(y), int(w), int(h)))
    return illo, boxes


def count_columns(text_mask, bbox):
    """Сколько колонок текста. Складываем чернила по вертикали: над колонками
    значения большие, над «улочками» между ними — маленькие. Колонки — это
    непрерывные участки, где чернил много."""
    x0, y0, x1, y1 = bbox
    region = text_mask[y0:y1, x0:x1]
    if region.size == 0:
        return 0, []
    profile = region.sum(axis=0) / 255.0
    if profile.max() <= 0:
        return 0, []
    # лёгкое сглаживание, чтобы профиль не «дрожал»
    smooth = cv2.GaussianBlur(profile.reshape(1, -1).astype(np.float32),
                              (0, 0), max(1, region.shape[1] // 120)).ravel()
    active = smooth > 0.12 * smooth.max()            # где колонка «активна»
    min_width = 0.04 * region.shape[1]               # колонка не уже 4% ширины
    cols = [(s + x0, e + x0) for (s, e) in _runs(active) if (e - s) >= min_width]

    # Защита от ложной колонки: крупный заголовок (например «СТИХИ») может
    # дать узкий столбец чернил без строк под ним. Оставляем только колонки,
    # в которых реально есть хотя бы 3 строки текста.
    if len(cols) > 1:
        real = []
        for (cx0, cx1) in cols:
            row_ink = text_mask[y0:y1, cx0:cx1].sum(axis=1) / 255.0
            if row_ink.max() <= 0:
                continue
            lines = _runs(row_ink > 0.15 * row_ink.max())
            if len(lines) >= 3:
                real.append((cx0, cx1))
        if real:
            cols = real
    return len(cols), cols


def count_text_lines(text_mask, col_span, y0, y1):
    """Сколько строк текста в колонке. Складываем чернила по горизонтали —
    получаются «пики» на строках; считаем пики."""
    x0, x1 = col_span
    region = text_mask[y0:y1, x0:x1]
    if region.size == 0:
        return 0
    profile = region.sum(axis=1) / 255.0
    if profile.max() <= 0:
        return 0
    peaks, _ = find_peaks(profile, height=0.20 * profile.max(), distance=6)
    return int(len(peaks))


def analyze_right_edge(text_mask, col_span, y0, y1, *, deskew=True):
    """Новый анализ одной колонки: строки, края, локальный наклон и диагностика.

    Короткие строки не отсеиваются по доле от самой длинной; одиночная пылинка
    не становится краем. Не используется для отбора текста по жанру.
    """
    from yunost.edges import measure_edges
    return measure_edges(text_mask, col_span, y0, y1, deskew=deskew)


def right_edge_raggedness(text_mask, col_span, y0, y1, *, method='legacy'):
    """std правых краёв / ширина колонки (не вероятность и не классификация).

    ``components-v2`` — новый компонентный анализ с локальным выравниванием.
    ``legacy`` — замороженная проекционная формула исходного исследования;
    оставлена по умолчанию для воспроизводимости ноутбука и старых CSV.
    Диагностические линии нового метода доступны через analyze_right_edge().
    """
    if method == 'components-v2':
        value = analyze_right_edge(text_mask, col_span, y0, y1)['value']
        return float('nan') if value is None else value
    if method != 'legacy':
        raise ValueError('Метод края: legacy или components-v2')
    x0, x1 = col_span
    region = text_mask[y0:y1, x0:x1]
    if region.size == 0:
        return np.nan
    width = region.shape[1]
    profile = region.sum(axis=1) / 255.0
    if profile.max() <= 0:
        return np.nan
    line_rows = np.where(profile > 0.20 * profile.max())[0]
    if len(line_rows) < 4:
        return np.nan
    # группируем соседние строки-пиксели в строки (разрыв > 3 px = новая строка)
    groups, current = [], [line_rows[0]]
    for r in line_rows[1:]:
        if r - current[-1] <= 3:
            current.append(r)
        else:
            groups.append(current)
            current = [r]
    groups.append(current)
    # правый край каждой строки (доля ширины колонки)
    edges = []
    for g in groups:
        band = region[g[0]:g[-1] + 1, :]
        ink_cols = np.where(band.sum(axis=0) > 0)[0]
        if len(ink_cols):
            edges.append(ink_cols.max() / width)
    if len(edges) < 4:
        return np.nan
    return float(np.std(edges))


def extract_features(path, *, raggedness_method='legacy'):
    """Признаки страницы; новый край включается явно, старые результаты не смешиваются.

    Только raggedness зависит от raggedness_method. Поиск колонок/иллюстраций —
    исторический; для сложной страницы надёжнее измерять отдельные текстовые блоки.
    """
    gray = load_gray(path)
    H, W = gray.shape
    page_area = H * W

    mask = ink_mask(gray)
    x0, y0, x1, y1 = content_bbox(mask)
    content_area = max(1, (x1 - x0) * (y1 - y0))

    # отделяем иллюстрации от текста
    illo, illo_boxes = detect_illustrations(mask)
    text_mask = cv2.bitwise_and(mask, cv2.bitwise_not(illo))

    ink_total = mask.sum() / 255.0
    ink_text = text_mask.sum() / 255.0
    illo_area = illo.sum() / 255.0

    n_cols, cols = count_columns(text_mask, (x0, y0, x1, y1))

    # «рваность» и число строк считаем по каждой колонке и берём медиану
    spans = cols if cols else [(x0, x1)]
    raggedness_values, line_counts = [], []
    for span in spans:
        rag = right_edge_raggedness(text_mask, span, y0, y1, method=raggedness_method)
        if not np.isnan(rag):
            raggedness_values.append(rag)
        line_counts.append(count_text_lines(text_mask, span, y0, y1))

    edge_density = cv2.Canny(gray, 60, 160).mean() / 255.0

    return dict(
        ink_density=ink_total / page_area,                  # доля чернил на странице
        text_density=ink_text / content_area,               # плотность текста в рамке
        illo_area_ratio=illo_area / page_area,              # доля площади иллюстраций
        n_illustrations=len(illo_boxes),
        whitespace_ratio=1 - content_area / page_area,      # доля полей («воздух»)
        n_columns=n_cols,
        line_count=int(np.median(line_counts)) if line_counts else 0,
        raggedness=float(np.median(raggedness_values)) if raggedness_values else np.nan,
        edge_density=float(edge_density),
    )
