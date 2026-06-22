from pathlib import Path
import math
import cv2
import numpy as np
from ultralytics import YOLO

# ---------- 配置 ----------
MODEL_PATH = r"D:\examples_project\ultralytics-main\runs\obb\train42\weights\best.pt"
SOURCE_DIR = r"G:\dataset\260622-obb-paper\test"
OUTPUT_DIR = r"G:\dataset\260622-obb-paper\out"

ROWS = 4
COLS = 6
TILE_W = 640
TILE_H = 640
IMGSZ = 1024
CONF = 0.5

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------- 工具函数 ----------
def fit_to_tile(image, tile_w, tile_h, bg_color=(255, 255, 255)):
    """等比例缩放图片并居中填充到固定瓦片尺寸。"""
    h, w = image.shape[:2]
    scale = min(tile_w / w, tile_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    tile = np.full((tile_h, tile_w, 3), bg_color, dtype=np.uint8)

    x = (tile_w - new_w) // 2
    y = (tile_h - new_h) // 2
    tile[y:y + new_h, x:x + new_w] = resized
    return tile

def make_sheet(images, rows, cols, tile_w, tile_h):
    """将一组已标注图片拼为固定行列的大图。"""
    capacity = rows * cols
    canvas = np.full((rows * tile_h, cols * tile_w, 3), 255, dtype=np.uint8)

    for index, image in enumerate(images[:capacity]):
        row, col = divmod(index, cols)
        tile = fit_to_tile(image, tile_w, tile_h)

        y1, y2 = row * tile_h, (row + 1) * tile_h
        x1, x2 = col * tile_w, (col + 1) * tile_w
        canvas[y1:y2, x1:x2] = tile

    return canvas

# ---------- 推理并拼图 ----------
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

image_files = sorted(
    p for p in Path(SOURCE_DIR).rglob("*")
    if p.suffix.lower() in IMAGE_EXTS
)

if not image_files:
    raise RuntimeError(f"没有在 {SOURCE_DIR} 找到图片")

model = YOLO(MODEL_PATH)
capacity = ROWS * COLS
page_images = []
page_index = 1

# stream=True 避免一次性把全部推理结果堆在内存里
for result in model(image_files, stream=True, imgsz=IMGSZ, conf=CONF):
    annotated = result.plot()  # 含 OBB 旋转框、类别和置信度，BGR 格式

    # 可选：左上角写入文件名
    name = Path(result.path).name
    cv2.putText(
        annotated, name, (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA
    )

    page_images.append(annotated)

    if len(page_images) == capacity:
        sheet = make_sheet(page_images, ROWS, COLS, TILE_W, TILE_H)
        save_path = output_dir / f"sheet_{page_index:03d}.jpg"
        cv2.imwrite(str(save_path), sheet)
        print(f"已输出: {save_path}")

        page_images = []
        page_index += 1

# 输出不足一页的剩余图片
if page_images:
    sheet = make_sheet(page_images, ROWS, COLS, TILE_W, TILE_H)
    save_path = output_dir / f"sheet_{page_index:03d}.jpg"
    cv2.imwrite(str(save_path), sheet)
    print(f"已输出: {save_path}")
