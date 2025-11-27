import os
import math
import openslide
import numpy as np
import matplotlib.pyplot as plt
import cv2

folder = "/gold/data_feasibility"
svs_files = [f for f in os.listdir(folder) if f.lower().endswith(".svs")]

level = 2

def compute_tissue_mask(img_rgb):
    img_np = np.array(img_rgb)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

n = len(svs_files)
cols = 4
rows = 3 * math.ceil(n / cols)  # 3 rows per slide

plt.figure(figsize=(5 * cols, 5 * rows))

for idx, filename in enumerate(svs_files):
    path = os.path.join(folder, filename)
    slide = openslide.OpenSlide(path)

    # low-res
    w, h = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, (w, h)).convert("RGB")

    # mask
    mask = compute_tissue_mask(img)
    img_np = np.array(img)
    img_masked = img_np.copy()
    img_masked[mask == 0] = 255

    # compute base row for this slide
    base_row = (idx // cols) * 3      # 3 rows per slide
    col = idx % cols

    # --- Plot low-res ---
    plt.subplot(rows, cols, base_row * cols + col + 1)
    plt.imshow(img)
    plt.title(f"{filename}\nLow-res", fontsize=9)
    plt.axis("off")

    # --- Plot mask ---
    plt.subplot(rows, cols, (base_row + 1) * cols + col + 1)
    plt.imshow(mask, cmap="gray")
    plt.title("Tissue mask", fontsize=9)
    plt.axis("off")

    # --- Plot masked image ---
    plt.subplot(rows, cols, (base_row + 2) * cols + col + 1)
    plt.imshow(img_masked)
    plt.title("Masked image", fontsize=9)
    plt.axis("off")

plt.tight_layout()
plt.savefig("/otsu/pannel.png", dpi=250, bbox_inches="tight")
