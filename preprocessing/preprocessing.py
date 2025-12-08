import os
import openslide
import numpy as np
import cv2
from PIL import Image
from skimage import transform
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.transform import AffineTransform
import warnings
import wandb
from tqdm import tqdm
warnings.filterwarnings('ignore')
from create_mask import compute_tissue_mask
from registration import register_whole_slide
import matplotlib.pyplot as plt
wandb.login(key="ab67e0f4c27fad7a0d47405f84a8a4deb80056ba")
# todo : enlever les régions et ne garder que les patches extraits

# Configuration
hes_slide_path = "/silver/ube/slides/c_HES.svs"
cd30_slide_path = "/silver/ube/slides/c_CD30.svs"
output_folder = "/silver/ube/extraction_v1"
patch_size = 2000
stride_patch = 1500
level = 2
tissue_threshold = 0.80


hes_dir = os.path.join(output_folder, "HES")
cd30_dir = os.path.join(output_folder, "CD30")
os.makedirs(hes_dir, exist_ok=True)
os.makedirs(cd30_dir, exist_ok=True)

wandb.init(
    project="ia2hl-preprocessing"
)

def patch_has_tissue(x, y, mask, downsample):
    x_lr = x // downsample
    y_lr = y // downsample
    ps_lr = patch_size // downsample
    patch_mask = mask[y_lr:y_lr+ps_lr, x_lr:x_lr+ps_lr]
    tissue_ratio = np.mean(patch_mask > 0)
    return tissue_ratio >= tissue_threshold

def process_slide_pair(hes_path, cd30_path, hes_dir, cd30_dir):
    slide_hes = openslide.OpenSlide(hes_path)
    slide_cd30 = openslide.OpenSlide(cd30_path)
    patient_id = os.path.splitext(os.path.basename(hes_path))[0].replace("_HES", "")
    patient_hes_dir = os.path.join(hes_dir, patient_id)
    patient_cd30_dir = os.path.join(cd30_dir, patient_id)
    os.makedirs(patient_hes_dir, exist_ok=True)
    os.makedirs(patient_cd30_dir, exist_ok=True)
    print("Chargement des images basse résolution...")
    w__hes, h_hes = slide_hes.level_dimensions[level]
    w_cd30, h_cd30 = slide_cd30.level_dimensions[level]
    lowres_hes = slide_hes.read_region((0, 0), level, (w__hes, h_hes)).convert("RGB")
    lowres_cd30 = slide_cd30.read_region((0, 0), level, (w_cd30, h_cd30)).convert("RGB")
    lowres_hes_np = np.array(lowres_hes)
    lowres_cd30_np = np.array(lowres_cd30)
    print(" Calcul du masque de tissu...")
    mask_hes = compute_tissue_mask(lowres_hes)
    model = register_whole_slide(lowres_hes_np, lowres_cd30_np, patient_id)
    total_patch_count = 0
    w0_hes, h0_hes = slide_hes.level_dimensions[0]
    w0_cd30, h0_cd30 = slide_cd30.level_dimensions[0]
    downsample_hes = int(slide_hes.level_downsamples[level])
    downsample_cd30 = int(slide_cd30.level_downsamples[level])

    # Calculer le nombre total d'itérations possibles
    y_positions = range(0, h0_hes, patch_size)
    x_positions = range(0, w0_hes, patch_size)
    total_iterations = len(list(y_positions)) * len(list(x_positions))
    
    # Créer la barre de progression
    pbar = tqdm(total=total_iterations, desc=f"Extraction patches {patient_id}", unit="patch")

    for y in range(0, h0_hes, patch_size):
        for x in range(0, w0_hes, patch_size):
            pbar.update(1)
            
            if x + patch_size > w0_hes or y + patch_size > h0_hes:
                continue
            if not patch_has_tissue(x, y, mask_hes, downsample_hes):
                continue
            patch_hes = slide_hes.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
            x_lr = x / downsample_hes
            y_lr = y / downsample_hes
            transform_matrix = model.params[:2]
            full_matrix = np.vstack([transform_matrix, [0, 0, 1]])
            inv_matrix = np.linalg.inv(full_matrix)
            point = np.array([x_lr, y_lr, 1])
            transformed_point = inv_matrix @ point
            x_cd30_lr = transformed_point[0]
            y_cd30_lr = transformed_point[1]
            x_cd30 = int(x_cd30_lr * downsample_cd30)
            y_cd30 = int(y_cd30_lr * downsample_cd30)
            if x_cd30 < 0 or y_cd30 < 0 or x_cd30 + patch_size > w0_cd30 or y_cd30 + patch_size > h0_cd30:
                continue
            patch_cd30 = slide_cd30.read_region((x_cd30, y_cd30), 0, (patch_size, patch_size)).convert("RGB")
            patch_name = f"patch_x{x}_y{y}.jpg"
            patch_hes.save(os.path.join(patient_hes_dir, patch_name), optimize=False)
            patch_cd30.save(os.path.join(patient_cd30_dir, patch_name), optimize=False)
            total_patch_count += 1
            pbar.set_postfix({"patches_extraits": total_patch_count})
    
    pbar.close()
    slide_hes.close()
    slide_cd30.close()
    return total_patch_count

patch_count = process_slide_pair(hes_slide_path, cd30_slide_path, hes_dir, cd30_dir)
print(f"Total: {patch_count} paires de patches extraites")
wandb.finish()
