import os
import math
import openslide
import numpy as np
import matplotlib.pyplot as plt
import cv2

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

if __name__ == "__main__":
    image_path = "/silver/ube/slides/e_HES.svs"
    level = 2

    slide = openslide.OpenSlide(image_path)
    w, h = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, (w, h)).convert("RGB")
    mask = compute_tissue_mask(img)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis("off")
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
    plt.title("Tissue Mask")

    plt.tight_layout()
    plt.savefig("/silver/ube/visualization/tissue_mask_single.png", dpi=500)