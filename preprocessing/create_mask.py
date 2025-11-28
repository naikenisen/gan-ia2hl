import os
import math
import openslide
import numpy as np
import matplotlib.pyplot as plt
import cv2

folder = "/tmp/data/gold"
svs_files = [f for f in os.listdir(folder) if f.lower().endswith("HES.svs")]
level = 2

def compute_tissue_mask(img_rgb):
    img_np = np.array(img_rgb)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    sat_eq = cv2.equalizeHist(sat)
    sat_blur = cv2.GaussianBlur(sat_eq, (15, 15), 2)
    _, mask = cv2.threshold(sat_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

n = len(svs_files)
cols = 4
rows = 2 * math.ceil(n / cols)

plt.figure(figsize=(5 * cols, 5 * rows))
for idx, filename in enumerate(svs_files):
    path = os.path.join(folder, filename)
    slide = openslide.OpenSlide(path)
    w, h = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, (w, h)).convert("RGB")
    mask = compute_tissue_mask(img)
    base_row = (idx // cols) * 2
    col = idx % cols
    plt.subplot(rows, cols, base_row * cols + col + 1)
    plt.imshow(img)
    plt.axis("off")
    plt.subplot(rows, cols, (base_row + 1) * cols + col + 1)
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

plt.tight_layout()
plt.savefig("tissue_masks.png", dpi=500)