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
warnings.filterwarnings('ignore')
from create_mask import compute_tissue_mask
from registration import register_whole_slide
import matplotlib.pyplot as plt
wandb.login(key="ab67e0f4c27fad7a0d47405f84a8a4deb80056ba")
# todo : enlever les régions et ne garder que les patches extraits

# Configuration
input_folder = "/gold/data_feasibility"
output_folder = "/silver/ube/patches"
patch_size = 2000
stride_patch = 1500
level = 2
tissue_threshold = 0.60
total_patches = 0
processed_slides = 0
failed_slides = 0
patients_to_process = ["AHL001", "AHL003", "AHL004",
                       "AHL006", "AHL007", "AHL011" ]


hes_dir = os.path.join(output_folder, "HES")
cd30_dir = os.path.join(output_folder, "CD30")
os.makedirs(hes_dir, exist_ok=True)
os.makedirs(cd30_dir, exist_ok=True)

wandb.init(
    project="ia2hl-preprocessing",
    config={
        "patch_size": patch_size,
        "stride_patch": stride_patch,
        "tissue_threshold": tissue_threshold,
        "level": level
    }
)

def patch_has_tissue(x, y, mask, downsample):
    x_lr = x // downsample
    y_lr = y // downsample
    ps_lr = patch_size // downsample
    patch_mask = mask[y_lr:y_lr+ps_lr, x_lr:x_lr+ps_lr]
    if patch_mask.size == 0:
        return False
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
    transformation = register_whole_slide(lowres_hes_np, lowres_cd30_np, patient_id)

    h, w = lowres_hes_np.shape[:2]
    aligned_cd30_global = cv2.warpAffine(
        lowres_cd30_np, 
        transformation.params[:2], 
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
    
    #stop code here
    return 0

    patch_count = 0
    total_patch_count = 0
    w0_hes, h0_hes = slide_hes.level_dimensions[0]
    w0_cd30, h0_cd30 = slide_cd30.level_dimensions[0]
    downsample_hes = int(slide_hes.level_downsamples[level])
    downsample_cd30 = int(slide_cd30.level_downsamples[level])

    for y in range(0, h0_hes, patch_size):
        for x in range(0, w0_hes, patch_size):
            if x + patch_size > w0_hes or y + patch_size > h0_hes:
                continue
            if not patch_has_tissue(x, y, mask_hes, downsample_hes):
                continue
            patch_hes = slide_hes.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
            x_lr = x / downsample_hes
            y_lr = y / downsample_hes
            transform_matrix = transformation.params[:2]
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
            patch_hes.save(os.path.join(patient_hes_dir, patch_name), optimize=True)
            patch_cd30.save(os.path.join(patient_cd30_dir, patch_name), optimize=True)
            patch_count += 1
            total_patch_count += 1
            print(f"{patch_count} paires de patches extraites")
    print(f"Total: {total_patch_count} paires de patches extraites")

    slide_hes.close()
    slide_cd30.close()
    return total_patch_count

svs_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".svs")]
hes_files = [f for f in svs_files if f.endswith("_HES.svs")]
cd30_files = [f for f in svs_files if f.endswith("_CD30.svs")]
print(f"\nFichiers trouvés: {len(hes_files)} HES, {len(cd30_files)} CD30")

pairs = {}
for hes_file in hes_files:
    base_id = hes_file.replace("_HES.svs", "")
    cd30_file = f"{base_id}_CD30.svs"
    for cd30_file in cd30_files:
        pairs[base_id] = {
            'hes': os.path.join(input_folder, hes_file),
            'cd30': os.path.join(input_folder, cd30_file)
        }
print(f"{len(pairs)} paires de slides à traiter")

for idx, (base_id, paths) in enumerate(pairs.items(), 1):
    if base_id not in patients_to_process:
        continue
    print(f"Paire {idx}/{len(pairs)}: {base_id}")
    patch_count = process_slide_pair(paths['hes'], paths['cd30'], hes_dir, cd30_dir)
    total_patches += patch_count
    processed_slides += 1
        
print("TRAITEMENT TERMINÉ")
print(f"Total: {total_patches} paires de patches extraites")
print(f"Lames traitées avec succès: {processed_slides}/{len(pairs)}")
print(f"Lames échouées: {failed_slides}")
print(f"Patches HES sauvegardés dans: {hes_dir}")
print(f"Patches CD30 sauvegardés dans: {cd30_dir}")
wandb.finish()
