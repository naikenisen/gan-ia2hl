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

# Configuration
input_folder = "/gold/data_feasibility"
output_folder = "/silver/ube/extract"
patch_size = 3000
region_size = 15000  # Taille des sous-régions (pour lames ~80000x70000)
stride_region = 15000  # Stride entre sous-régions (non-overlap si = region_size)
lowres_level = 2
tissue_threshold = 0.80
jpeg_quality = 90  # Qualité JPEG pour compression

hes_dir = os.path.join(output_folder, "HES")
cd30_dir = os.path.join(output_folder, "CD30")
os.makedirs(hes_dir, exist_ok=True)
os.makedirs(cd30_dir, exist_ok=True)

# Initialisation wandb
wandb.login(key="ab67e0f4c27fad7a0d47405f84a8a4deb80056ba")
wandb.init(
    project="ia2hl-preprocessing",
    config={
        "patch_size": patch_size,
        "region_size": region_size,
        "stride_region": stride_region,
        "tissue_threshold": tissue_threshold,
        "jpeg_quality": jpeg_quality,
        "lowres_level": lowres_level
    }
)


def compute_tissue_mask(img_rgb):
    """Calcule un masque binaire des tissus basé sur la saturation."""
    img_np = np.array(img_rgb)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]

    _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def align_images_orb(img_fixed, img_moving):
    """
    Aligne deux images en utilisant ORB + RANSAC.
    
    Args:
        img_fixed: Image de référence (RGB numpy array)
        img_moving: Image à aligner (RGB numpy array)
    
    Returns:
        transformation: Matrice de transformation affine
        aligned_img: Image alignée
        success: Booléen indiquant le succès de l'alignement
    """
    # Conversion en niveaux de gris
    gray_fixed = cv2.cvtColor(img_fixed, cv2.COLOR_RGB2GRAY)
    gray_moving = cv2.cvtColor(img_moving, cv2.COLOR_RGB2GRAY)
    
    # Détection de features avec ORB
    orb = cv2.ORB_create(nfeatures=5000)
    
    kp1, desc1 = orb.detectAndCompute(gray_fixed, None)
    kp2, desc2 = orb.detectAndCompute(gray_moving, None)
    
    if desc1 is None or desc2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("  ⚠ Pas assez de points détectés pour l'alignement")
        return None, img_moving, False
    
    # Matching des descripteurs
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Garder les meilleurs matches
    num_good_matches = min(len(matches), max(50, int(len(matches) * 0.15)))
    good_matches = matches[:num_good_matches]
    
    if len(good_matches) < 4:
        print("  ⚠ Pas assez de correspondances pour l'alignement")
        return None, img_moving, False
    
    # Extraire les coordonnées des points correspondants
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    
    # Estimation de la transformation affine avec RANSAC
    try:
        model, inliers = ransac(
            (src_pts, dst_pts),
            AffineTransform,
            min_samples=3,
            residual_threshold=5.0,
            max_trials=1000
        )
        
        if model is None or np.sum(inliers) < 4:
            print(f"  ⚠ RANSAC échoué (inliers: {np.sum(inliers) if inliers is not None else 0})")
            return None, img_moving, False
        
        print(f"  ✓ Alignement réussi avec {np.sum(inliers)} inliers sur {len(good_matches)} matches")
        
        # Appliquer la transformation
        h, w = img_fixed.shape[:2]
        aligned_img = cv2.warpAffine(
            img_moving, 
            model.params[:2], 
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        
        return model, aligned_img, True
        
    except Exception as e:
        print(f"  ⚠ Erreur lors de l'alignement: {e}")
        return None, img_moving, False


def patch_has_tissue(x, y, mask, downsample):
    """Vérifie si un patch contient suffisamment de tissu."""
    x_lr = x // downsample
    y_lr = y // downsample
    ps_lr = patch_size // downsample

    patch_mask = mask[y_lr:y_lr+ps_lr, x_lr:x_lr+ps_lr]

    if patch_mask.size == 0:
        return False

    tissue_ratio = np.mean(patch_mask > 0)
    return tissue_ratio >= tissue_threshold


def process_slide_pair(hes_path, cd30_path, hes_dir, cd30_dir):
    """
    Traite une paire de slides HES/CD30 avec alignement par sous-région.
    """
    print(f"\n{'='*80}")
    print(f"Traitement de la paire:")
    print(f"  HES:  {os.path.basename(hes_path)}")
    print(f"  CD30: {os.path.basename(cd30_path)}")
    print(f"{'='*80}")
    
    # Ouvrir les slides
    slide_hes = openslide.OpenSlide(hes_path)
    slide_cd30 = openslide.OpenSlide(cd30_path)
    
    # Extraire les IDs (nom du patient)
    patient_id = os.path.splitext(os.path.basename(hes_path))[0].replace("_HES", "")
    
    # Créer les répertoires de sortie pour ce patient
    patient_hes_dir = os.path.join(hes_dir, patient_id)
    patient_cd30_dir = os.path.join(cd30_dir, patient_id)
    os.makedirs(patient_hes_dir, exist_ok=True)
    os.makedirs(patient_cd30_dir, exist_ok=True)
    
    # Étape 1: Charger les images basse résolution
    print("\n[1/4] Chargement des images basse résolution...")
    w_lr_hes, h_lr_hes = slide_hes.level_dimensions[lowres_level]
    w_lr_cd30, h_lr_cd30 = slide_cd30.level_dimensions[lowres_level]

    lowres_hes = slide_hes.read_region(
        (0, 0), lowres_level, (w_lr_hes, h_lr_hes)
    ).convert("RGB")

    lowres_cd30 = slide_cd30.read_region(
        (0, 0), lowres_level, (w_lr_cd30, h_lr_cd30)
    ).convert("RGB")

    lowres_hes_np = np.array(lowres_hes)
    lowres_cd30_np = np.array(lowres_cd30)

    # Étape 2: Calcul du masque de tissu
    print("\n[2/4] Calcul du masque de tissu...")
    mask_hes = compute_tissue_mask(lowres_hes)

    # Étape 3: Découpe en sous-régions et alignement
    print("\n[3/4] Découpe en sous-régions et alignement...")
    w0_hes, h0_hes = slide_hes.level_dimensions[0]
    w0_cd30, h0_cd30 = slide_cd30.level_dimensions[0]
    downsample_hes = int(slide_hes.level_downsamples[lowres_level])
    downsample_cd30 = int(slide_cd30.level_downsamples[lowres_level])

    total_patch_count = 0
    region_index = 0

    for region_y in range(0, h0_hes, stride_region):
        for region_x in range(0, w0_hes, stride_region):
            # Vérifier que la région ne dépasse pas les limites
            if region_x + region_size > w0_hes or region_y + region_size > h0_hes:
                continue
            
            # Vérifier si la région contient suffisamment de tissu
            region_x_lr = int(region_x / downsample_hes)
            region_y_lr = int(region_y / downsample_hes)
            region_size_lr = int(region_size / downsample_hes)
            
            region_mask = mask_hes[region_y_lr:region_y_lr+region_size_lr, region_x_lr:region_x_lr+region_size_lr]
            if region_mask.size == 0 or np.mean(region_mask > 0) < tissue_threshold:
                continue
            
            # Extraire les régions basse résolution pour alignement
            region_hes_lr = lowres_hes_np[region_y_lr:region_y_lr+region_size_lr, region_x_lr:region_x_lr+region_size_lr]
            
            # Coordonnées correspondantes dans CD30
            region_x_cd30_lr = int(region_x / downsample_cd30)
            region_y_cd30_lr = int(region_y / downsample_cd30)
            region_size_cd30_lr = int(region_size / downsample_cd30)
            
            if region_x_cd30_lr + region_size_cd30_lr > w_lr_cd30 or region_y_cd30_lr + region_size_cd30_lr > h_lr_cd30:
                continue
            
            region_cd30_lr = lowres_cd30_np[region_y_cd30_lr:region_y_cd30_lr+region_size_cd30_lr, region_x_cd30_lr:region_x_cd30_lr+region_size_cd30_lr]

            # Alignement local sur la région
            print(f"\n  Région {region_index}: x={region_x}, y={region_y}")
            transformation, _, success = align_images_orb(region_hes_lr, region_cd30_lr)
            
            if not success:
                print(f"    ⚠ Alignement échoué, région ignorée")
                continue

            # Créer les sous-dossiers pour cette sous-région
            subregion_name = f"region_{region_index:03d}_x{region_x}_y{region_y}"
            subregion_hes_dir = os.path.join(patient_hes_dir, subregion_name)
            subregion_cd30_dir = os.path.join(patient_cd30_dir, subregion_name)
            os.makedirs(subregion_hes_dir, exist_ok=True)
            os.makedirs(subregion_cd30_dir, exist_ok=True)

            # Étape 4: Extraction des patchs dans la région alignée
            patch_count = 0
            for y in range(region_y, region_y + region_size, patch_size):
                for x in range(region_x, region_x + region_size, patch_size):
                    if x + patch_size > region_x + region_size or y + patch_size > region_y + region_size:
                        continue
                    
                    # Vérifier la présence de tissu
                    if not patch_has_tissue(x, y, mask_hes, downsample_hes):
                        continue
                    
                    # Extraire le patch HES
                    patch_hes = slide_hes.read_region(
                        (x, y), 0, (patch_size, patch_size)
                    ).convert("RGB")
                    
                    # Appliquer la transformation locale aux coordonnées du patch
                    x_local = x - region_x
                    y_local = y - region_y
                    x_local_lr = x_local / downsample_hes
                    y_local_lr = y_local / downsample_hes
                    
                    try:
                        transform_matrix = transformation.params[:2]
                        full_matrix = np.vstack([transform_matrix, [0, 0, 1]])
                        inv_matrix = np.linalg.inv(full_matrix)
                        point = np.array([x_local_lr, y_local_lr, 1])
                        transformed_point = inv_matrix @ point
                        
                        x_cd30_local_lr = transformed_point[0]
                        y_cd30_local_lr = transformed_point[1]
                        
                        x_cd30 = int(region_x + x_cd30_local_lr * downsample_cd30)
                        y_cd30 = int(region_y + y_cd30_local_lr * downsample_cd30)
                        
                    except Exception as e:
                        print(f"    ⚠ Erreur transformation: {e}")
                        x_cd30, y_cd30 = x, y
                    
                    # Vérifier les limites pour CD30
                    if x_cd30 < 0 or y_cd30 < 0 or x_cd30 + patch_size > w0_cd30 or y_cd30 + patch_size > h0_cd30:
                        continue
                    
                    # Extraire le patch CD30
                    patch_cd30 = slide_cd30.read_region(
                        (x_cd30, y_cd30), 0, (patch_size, patch_size)
                    ).convert("RGB")
                    
                    # Nommer les patches selon leurs coordonnées (relatives à la région)
                    patch_name = f"patch_x{x_local}_y{y_local}.jpg"
                    
                    # Sauvegarder avec compression JPEG
                    patch_hes.save(os.path.join(subregion_hes_dir, patch_name), 
                                   quality=jpeg_quality, optimize=True)
                    patch_cd30.save(os.path.join(subregion_cd30_dir, patch_name), 
                                    quality=jpeg_quality, optimize=True)
                    
                    patch_count += 1
            
            print(f"    ✓ {patch_count} paires de patches extraites")
            total_patch_count += patch_count
            
            # Log wandb pour cette région
            wandb.log({
                f"{patient_id}_region_{region_index}_patches": patch_count,
                f"{patient_id}_region_x": region_x,
                f"{patient_id}_region_y": region_y
            })
            
            region_index += 1
    
    print(f"\n✓ Total: {total_patch_count} paires de patches extraites dans {region_index} sous-régions")
    
    # Log wandb pour ce patient
    wandb.log({
        f"{patient_id}_total_patches": total_patch_count,
        f"{patient_id}_total_regions": region_index
    })
    
    # Fermer les slides
    slide_hes.close()
    slide_cd30.close()
    
    return total_patch_count


# ============================================================================
# EXÉCUTION PRINCIPALE
# ============================================================================

print(f"\n{'='*80}")
print("ALIGNEMENT ET DÉCOUPE DE PATCHES POUR PIX2PIX")
print(f"{'='*80}")
print(f"Dossier d'entrée: {input_folder}")
print(f"Dossier de sortie: {output_folder}")
print(f"Taille des patches: {patch_size}x{patch_size}")
print(f"Seuil de tissu: {tissue_threshold}")

# Lister tous les fichiers SVS
svs_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".svs")]

# Séparer HES et CD30
hes_files = [f for f in svs_files if f.endswith("_HES.svs")]
cd30_files = [f for f in svs_files if f.endswith("_CD30.svs")]

print(f"\nFichiers trouvés: {len(hes_files)} HES, {len(cd30_files)} CD30")

# Créer un dictionnaire pour apparier les slides
pairs = {}
for hes_file in hes_files:
    # Extraire l'ID de base (sans _HES.svs)
    base_id = hes_file.replace("_HES.svs", "")
    cd30_file = f"{base_id}_CD30.svs"
    
    if cd30_file in cd30_files:
        pairs[base_id] = {
            'hes': os.path.join(input_folder, hes_file),
            'cd30': os.path.join(input_folder, cd30_file)
        }
    else:
        print(f"⚠ Pas de correspondance CD30 pour {hes_file}")

print(f"\n{len(pairs)} paires de slides à traiter\n")

# Traiter chaque paire
total_patches = 0
processed_slides = 0
failed_slides = 0

for idx, (base_id, paths) in enumerate(pairs.items(), 1):
    print(f"\n{'*'*80}")
    print(f"Paire {idx}/{len(pairs)}: {base_id}")
    print(f"{'*'*80}")
    
    try:
        patch_count = process_slide_pair(paths['hes'], paths['cd30'], hes_dir, cd30_dir)
        total_patches += patch_count
        processed_slides += 1
        
        # Log wandb progression globale
        wandb.log({
            "total_patches_so_far": total_patches,
            "processed_slides": processed_slides,
            "progress_pct": (idx / len(pairs)) * 100
        })
        
    except Exception as e:
        print(f"\n✗ Erreur lors du traitement de {base_id}: {e}")
        import traceback
        traceback.print_exc()
        failed_slides += 1
        wandb.log({"failed_slides": failed_slides})
        continue

print(f"\n{'='*80}")
print("TRAITEMENT TERMINÉ")
print(f"{'='*80}")
print(f"Total: {total_patches} paires de patches extraites")
print(f"Lames traitées avec succès: {processed_slides}/{len(pairs)}")
print(f"Lames échouées: {failed_slides}")
print(f"Patches HES sauvegardés dans: {hes_dir}")
print(f"Patches CD30 sauvegardés dans: {cd30_dir}")

# Log wandb final
wandb.log({
    "final_total_patches": total_patches,
    "final_processed_slides": processed_slides,
    "final_failed_slides": failed_slides
})

# Terminer wandb
wandb.finish()
