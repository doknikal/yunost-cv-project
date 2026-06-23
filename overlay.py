"""
overlay.py — находит СТРОФЫ на страницах журнала «Юность 1962 №12».

Что делает: берёт фотографии страниц из папки, убирает иллюстрации (через
cvpipe), затем «склеивает» строки в строфы и считает их. Рядом подписывает,
сколько колонок на странице. Для каждой страницы сохраняет картинку с рамками
строф и общую таблицу stanzas_1962-12.csv.

Как находим строфы (классический CV, без нейросетей):
  1) горизонтально «склеиваем» слова в сплошные строки-полоски;
  2) вертикально склеиваем строки, стоящие близко (внутри одной строфы);
     увеличенный пробел между строфами при этом остаётся разрывом;
  3) каждое получившееся отдельное пятно = одна строфа.

Запуск:
    python overlay.py                       # папка по умолчанию (в Загрузках)
    python overlay.py "/путь/к/папке/со/страницами"
"""
import os
import sys
import glob
import csv
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")   # печать кириллицы в консоль Windows (cp1252)

import cv2
import numpy as np
import pillow_heif
from PIL import Image, ImageDraw, ImageFont

import cvpipe

pillow_heif.register_heif_opener()      # чтобы PIL умел открывать .heic с iPhone

# --- где брать страницы и куда складывать результат ---
# Исходные страницы номера 1962 №12 лежат в папке 1962/ рядом со скриптом
DEFAULT_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1962")
SRC = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_1962-12")
TARGET_HEIGHT = 2000                    # приводим все страницы к одной высоте


def find_font():
    """Ищем шрифт с кириллицей (чтобы подпись не превратилась в «?????»)."""
    for path in ["C:/Windows/Fonts/arial.ttf",                     # Windows
                 "/System/Library/Fonts/Supplemental/Arial.ttf",   # macOS
                 "/Library/Fonts/Arial.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:  # Linux
        if os.path.exists(path):
            return ImageFont.truetype(path, 42)
    return ImageFont.load_default()


FONT = find_font()


def draw_label(image_bgr, text):
    """Пишем русский текст вверху слева. cv2 кириллицу не умеет, поэтому
    рисуем через PIL и возвращаем картинку обратно в формат OpenCV."""
    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text((22, 16), text, fill=(220, 0, 0), font=FONT)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def morph_close(image, width, height):
    """Морфологическое «закрытие» прямоугольным ядром — склеивает близкие
    чёрные пиксели по горизонтали (width) и/или вертикали (height)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, height))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def detect_stanzas(path):
    """Находим строфы. Возвращаем список рамок (x, y, w, h) и словарь meta."""
    gray = cvpipe.load_gray(path)
    mask = cvpipe.ink_mask(gray)
    x0, y0, x1, y1 = cvpipe.content_bbox(mask)
    illo, illo_boxes = cvpipe.detect_illustrations(mask)
    text = cv2.bitwise_and(mask, cv2.bitwise_not(illo))   # убрали иллюстрации
    region = text[y0:y1, x0:x1]
    height, width = region.shape

    # оценим высоту строки: склеим слова в полоски и возьмём медиану их высот
    bars0 = morph_close(region, 15, 1)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bars0, 8)
    line_heights = [stats[i][3] for i in range(1, n)
                    if stats[i][2] > 0.08 * width and 3 < stats[i][3] < 0.05 * height]
    line_h = int(np.median(line_heights)) if line_heights else 14

    # шаг 1+2: склеиваем слова в строки, потом строки внутри строфы
    bars = morph_close(region, max(15, int(1.4 * line_h)), 1)
    blocks = morph_close(bars, 1, max(2, int(round(0.75 * line_h))))

    # шаг 3: каждое отдельное пятно — строфа (с фильтром мелкого мусора и плашек)
    n2, _, stats2, _ = cv2.connectedComponentsWithStats(blocks, 8)
    boxes = []
    for i in range(1, n2):
        x, y, w, h, _ = stats2[i]
        too_small = w < 0.10 * width or h < 0.6 * line_h
        full_width_title = w > 0.80 * width and h < 1.6 * line_h
        if too_small or full_width_title:
            continue
        boxes.append((x + x0, y + y0, w, h))

    meta = dict(content=(x0, y0, x1, y1), content_w=(x1 - x0),
                n_illustrations=len(illo_boxes))
    return boxes, meta


def count_side_by_side(spans, min_gap):
    """Сколько раздельных групп образуют горизонтальные отрезки (x, width).
    Отрезки идут слева направо; новый разрыв больше min_gap = новая колонка."""
    if not spans:
        return 0
    groups = 1
    right_end = spans[0][0] + spans[0][1]
    for x, w in spans[1:]:
        if x > right_end + min_gap:
            groups += 1
        right_end = max(right_end, x + w)
    return groups


def estimate_columns(boxes, content_w):
    """Сколько колонок на странице. Для каждой строфы смотрим, сколько строф
    стоят рядом с ней на той же высоте — это локальное число колонок. Берём
    самое частое значение по странице (моду), чтобы случайный сбой не мешал.
    Полноширинные заголовки в подсчёте не участвуют."""
    blocks = [b for b in boxes if b[2] < 0.7 * content_w] or list(boxes)
    if not blocks:
        return 0
    min_gap = 0.006 * content_w
    local_counts = []
    for _, y, _, h in blocks:
        y_mid = y + h / 2.0
        same_row = sorted((bx, bw) for bx, by, bw, bh in blocks
                          if by <= y_mid <= by + bh)
        local_counts.append(count_side_by_side(same_row, min_gap))
    freq = Counter(local_counts)
    most_common = max(freq.values())
    return max(n for n, c in freq.items() if c == most_common)


def make_overlay(path, out_path):
    """Рисуем рамки строф на странице и подписываем числа. Возвращаем
    (число строф, число колонок, число иллюстраций)."""
    boxes, meta = detect_stanzas(path)
    n_cols = estimate_columns(boxes, meta["content_w"])

    vis = cv2.cvtColor(cvpipe.load_gray(path), cv2.COLOR_GRAY2BGR)
    x0, y0, x1, y1 = meta["content"]
    cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 140, 0), 2)   # рамка текста
    colors = np.random.default_rng(1)
    for x, y, w, h in boxes:
        color = tuple(int(c) for c in colors.integers(60, 235, 3))
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 3)
    vis = draw_label(vis, f"строф: {len(boxes)}   колонок: {n_cols}")
    cv2.imwrite(out_path, vis)
    return len(boxes), n_cols, meta["n_illustrations"]


def normalize_page(src_path, dst_path):
    """Открываем страницу (в т.ч. .heic), приводим к высоте TARGET_HEIGHT, сохраняем PNG."""
    img = Image.open(src_path).convert("RGB")
    new_width = round(img.width * TARGET_HEIGHT / img.height)
    img.resize((new_width, TARGET_HEIGHT), Image.LANCZOS).save(dst_path)


def main():
    if not os.path.isdir(SRC):
        sys.exit(f"Папка не найдена: {SRC}")
    norm_dir = os.path.join(OUT, "norm")
    os.makedirs(norm_dir, exist_ok=True)

    files = sorted(f for f in glob.glob(os.path.join(SRC, "*"))
                   if f.lower().endswith((".heic", ".png", ".jpg", ".jpeg")))
    if not files:
        sys.exit(f"В папке нет изображений: {SRC}")

    rows = []
    for i, src in enumerate(files, start=1):
        norm_path = os.path.join(norm_dir, f"p{i:02d}.png")
        normalize_page(src, norm_path)
        overlay_path = os.path.join(OUT, f"p{i:02d}_stanzas.png")
        n_stanzas, n_cols, n_illo = make_overlay(norm_path, overlay_path)
        rows.append(dict(page=i, src=os.path.basename(src), n_stanzas=n_stanzas,
                         n_columns=n_cols, n_illustrations=n_illo))
        print(f"  стр.{i:02d}: строф={n_stanzas}, колонок={n_cols}, иллюстраций={n_illo}")

    with open(os.path.join(OUT, "stanzas_1962-12.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nГотово: {len(rows)} страниц. Результаты в папке {OUT}/")


if __name__ == "__main__":
    main()
