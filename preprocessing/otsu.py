import os
import math
import openslide
import numpy as np
import matplotlib.pyplot as plt
import cv2

folder = "/tmp/data/gold"
svs_files = [f for f in os.listdir(folder) if f.lower().endswith(".svs")]

level = 2

def compute_tissue_mask(img_rgb):
    img_np = np.array(img_rgb)

    # Conversion RGB -> HSV
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    # Normalisation de l'histogramme sur la saturation
    sat = hsv[:, :, 1]
    sat_eq = cv2.equalizeHist(sat)

    # --- Ajout du flou gaussien AVANT le seuillage ---
    sat_blur = cv2.GaussianBlur(sat_eq, (15, 15), 2)

    # Seuillage (Otsu) directement sur l'image floutée
    _, mask = cv2.threshold(sat_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphologie
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


n = len(svs_files)
cols = 4
rows = 2 * math.ceil(n / cols)  # 2 rows per slide

plt.figure(figsize=(5 * cols, 5 * rows))

for idx, filename in enumerate(svs_files):
    path = os.path.join(folder, filename)
    slide = openslide.OpenSlide(path)

    # low-res
    w, h = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, (w, h)).convert("RGB")

    # mask
    mask = compute_tissue_mask(img)

    # compute base row for this slide
    base_row = (idx // cols) * 2      # 2 rows per slide
    col = idx % cols

    # --- Plot low-res ---
    plt.subplot(rows, cols, base_row * cols + col + 1)
    plt.imshow(img)
    plt.axis("off")

    # --- Plot mask ---
    plt.subplot(rows, cols, (base_row + 1) * cols + col + 1)
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

plt.tight_layout()
plt.show()