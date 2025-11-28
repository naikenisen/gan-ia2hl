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
import wandb
import pandas as pd
from collections import defaultdict
warnings.filterwarnings('ignore')
wandb.login(key="ab67e0f4c27fad7a0d47405f84a8a4deb80056ba")

# Configuration
input_folder = "/gold/data_feasibility"
output_folder = "./visualization"
os.makedirs(output_folder, exist_ok=True)
registration_metrics = defaultdict(lambda: defaultdict(dict))
lowres_level = 2
minimal_paired_points = 3
maximal_error_threshold = 8.0
ransac_iterations = 2000

wandb.init(
    project="ia2hl-preprocessing",
    name="global-registration",
    config={
        "lowres_level": lowres_level,
        "input_folder": input_folder,
        "output_folder": output_folder,
        "minimal_paired_points": minimal_paired_points,
        "maximal_error_threshold": maximal_error_threshold,
        "ransac_iterations": ransac_iterations
    }
)

def register_whole_slide(lowres_hes_np, lowres_cd30_np, patient_id):
    """
    Effectue une registration globale de la lame entière à basse résolution.
    Retourne la transformation globale et l'image CD30 alignée.
    """
    # Conversion en niveaux de gris
    gray_hes = cv2.cvtColor(lowres_hes_np, cv2.COLOR_RGB2GRAY)
    gray_cd30 = cv2.cvtColor(lowres_cd30_np, cv2.COLOR_RGB2GRAY)
    # Détection AKAZE
    akaze = cv2.AKAZE_create()
    kp1, desc1 = akaze.detectAndCompute(gray_hes, None)
    kp2, desc2 = akaze.detectAndCompute(gray_cd30, None)
    print(f"Points détectés - HES: {len(kp1) if kp1 else 0}, CD30: {len(kp2) if kp2 else 0}")
    # Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    # AKAZE produit généralement plus de matches de qualité, on peut être plus sélectif
    num_good_matches = min(len(matches), max(100, int(len(matches) * 0.25)))
    good_matches = matches[:num_good_matches]
    print(f"Correspondances: {len(matches)} total, {len(good_matches)} sélectionnées")
    if len(good_matches) < 4:
        print("Pas assez de correspondances pour la registration globale")
        return None, lowres_cd30_np
    # Extraire les points
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    # RANSAC pour la transformation globale
    model_global, inliers = ransac(
        (src_pts, dst_pts),
        AffineTransform,
        min_samples=minimal_paired_points,
        residual_threshold= maximal_error_threshold,
        max_trials= ransac_iterations
    )
    if model_global is None or np.sum(inliers) < 4:
        print("RANSAC failed")
        return None, lowres_cd30_np
    num_inliers = np.sum(inliers)
    inlier_ratio = num_inliers / len(good_matches)
    print(f"Registration done: {num_inliers}/{len(good_matches)} inliers (ratio: {inlier_ratio:.3f})")
    return model_global

def build_figure(lowres_hes_np, lowres_cd30_np, patient_id, model):
    # Appliquer la transformation globale
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
    output_path = os.path.join(output_folder, f'{patient_id}_0_global_registration.png')
    plt.savefig(output_path, dpi=500, bbox_inches='tight')
    plt.close()
    print(f"Visualisation sauvegardée")


def process_one_slide_pair_visualization(hes_path, cd30_path):
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
    if model is None:
        return
    build_figure(lowres_hes_np, lowres_cd30_np, patient_id, model)
            
    slide_hes.close()
    slide_cd30.close()


svs_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".svs")]
hes_files = [f for f in svs_files if f.endswith("_HES.svs")]
cd30_files = [f for f in svs_files if f.endswith("_CD30.svs")]

print(f"\nFichiers trouvés: {len(hes_files)} HES, {len(cd30_files)} CD30")

# Créer les paires
pairs = {}
for hes_file in hes_files:
    base_id = hes_file.replace("_HES.svs", "")
    cd30_file = f"{base_id}_CD30.svs"
    
    if cd30_file in cd30_files:
        pairs[base_id] = {
            'hes': os.path.join(input_folder, hes_file),
            'cd30': os.path.join(input_folder, cd30_file)
        }

print(f"{len(pairs)} paires disponibles")

total_patients = len(pairs)
patients_with_regions = 0
total_regions = 0

print(f"Traitement de {total_patients} patients")

for idx, (patient_id, paths) in enumerate(pairs.items(), 1):
    print(f"\n[Patient {idx}/{total_patients}] Traitement de: {patient_id}")
    process_one_slide_pair_visualization(paths['hes'], paths['cd30'])

print(f"TRAITEMENT TERMINÉ")
wandb.finish()