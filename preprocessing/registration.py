import os
import openslide
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.measure import ransac
from skimage.transform import AffineTransform
import warnings
import pandas as pd
from collections import defaultdict
import random

def color_traitement(image):
    # amélioration du contraste et conversion en niveaux de gris
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.equalizeHist(s)
    hsv_enhanced = cv2.merge([h, s, v])
    rgb_enhanced = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)
    for i in range(3):
        rgb_enhanced[..., i] = cv2.equalizeHist(rgb_enhanced[..., i])
    gray = cv2.cvtColor(rgb_enhanced, cv2.COLOR_RGB2GRAY)
    return gray

def register_whole_slide(lowres_hes_np, lowres_cd30_np, patient_id, seed=42):
    # fonction principale de registration
    random.seed(seed)
    np.random.seed(seed)
    minimal_paired_points = 3
    maximal_error_threshold = 8.0
    ransac_iterations = 2000
    gray_hes = color_traitement(lowres_hes_np)
    gray_cd30 = color_traitement(lowres_cd30_np)
    akaze = cv2.AKAZE_create()
    kp1, desc1 = akaze.detectAndCompute(gray_hes, None)
    kp2, desc2 = akaze.detectAndCompute(gray_cd30, None)
    print(f"Points détectés - HES: {len(kp1) if kp1 else 0}, CD30: {len(kp2) if kp2 else 0}")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    num_good_matches = min(len(matches), max(100, int(len(matches) * 0.25)))
    good_matches = matches[:num_good_matches]
    print(f"Correspondances: {len(matches)} total, {len(good_matches)} sélectionnées")
    if len(good_matches) < 4:
        raise RuntimeError(f"registration failed, not enough matches")
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    model_global, inliers = ransac(
        (src_pts, dst_pts),
        AffineTransform,
        min_samples=minimal_paired_points,
        residual_threshold= maximal_error_threshold,
        max_trials= ransac_iterations
    )
    num_inliers = np.sum(inliers)
    if num_inliers < 4:
        raise RuntimeError(f"RANSAC failed, not enough inliers")
    inlier_ratio = num_inliers / len(good_matches)
    print(f"Registration done: {num_inliers}/{len(good_matches)} inliers (ratio: {inlier_ratio:.3f})")
    return model_global

def build_figure(lowres_hes_np, lowres_cd30_np, patient_id, model):
    # construction de la figure de verification pour allignement global
    h, w = lowres_hes_np.shape[:2]
    aligned_cd30_global = cv2.warpAffine(
        lowres_cd30_np, 
        model.params[:2], 
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    axes[0].imshow(lowres_hes_np)
    axes[0].axis('off')
    axes[1].imshow(lowres_cd30_np)
    axes[1].axis('off')
    axes[2].imshow(aligned_cd30_global)
    axes[2].axis('off')
    plt.suptitle(f'{patient_id}')
    plt.tight_layout()
    plt.savefig(f'/silver/ube/visualization/{patient_id}_registration.png', dpi=500, bbox_inches='tight')
    plt.close()

def process_one_slide_pair_visualization(hes_path, cd30_path):
    lowres_level = 2
    slide_hes = openslide.OpenSlide(hes_path)
    slide_cd30 = openslide.OpenSlide(cd30_path)
    patient_id = os.path.splitext(os.path.basename(hes_path))[0].replace("_HES", "")
    print("Chargement des images basse résolution")
    w_lr_hes, h_lr_hes = slide_hes.level_dimensions[lowres_level]
    w_lr_cd30, h_lr_cd30 = slide_cd30.level_dimensions[lowres_level]
    lowres_hes = slide_hes.read_region((0, 0), lowres_level, (w_lr_hes, h_lr_hes)).convert("RGB")
    lowres_cd30 = slide_cd30.read_region((0, 0), lowres_level, (w_lr_cd30, h_lr_cd30)).convert("RGB")
    lowres_hes_np = np.array(lowres_hes)
    lowres_cd30_np = np.array(lowres_cd30)
    model = register_whole_slide(lowres_hes_np, lowres_cd30_np, patient_id)
    build_figure(lowres_hes_np, lowres_cd30_np, patient_id, model)
    slide_hes.close()
    slide_cd30.close()

if __name__ == "__main__":
    hes_path = "/silver/ube/slides/a_HES.svs"
    cd30_path = "/silver/ube/slides/a_CD30.svs"
    output_folder = "/silver/ube/visualization"
    process_one_slide_pair_visualization(hes_path, cd30_path)