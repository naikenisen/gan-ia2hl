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
warnings.filterwarnings('ignore')

# Configuration
input_folder = "/gold/data_feasibility"
output_folder = "/silver/ube/data_feasibility"
patch_size = 512
lowres_level = 2
tissue_threshold = 0.60

hes_dir = os.path.join(output_folder, "HES")
cd30_dir = os.path.join(output_folder, "CD30")
os.makedirs(hes_dir, exist_ok=True)
os.makedirs(cd30_dir, exist_ok=True)


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
    Traite une paire de slides HES/CD30 avec alignement.
    """
    print(f"\n{'='*80}")
    print(f"Traitement de la paire:")
    print(f"  HES:  {os.path.basename(hes_path)}")
    print(f"  CD30: {os.path.basename(cd30_path)}")
    print(f"{'='*80}")
    
    # Ouvrir les slides
    slide_hes = openslide.OpenSlide(hes_path)
    slide_cd30 = openslide.OpenSlide(cd30_path)
    
    # Extraire les IDs
    hes_id = os.path.splitext(os.path.basename(hes_path))[0]
    cd30_id = os.path.splitext(os.path.basename(cd30_path))[0]
    
    # Créer les répertoires de sortie
    hes_out = os.path.join(hes_dir, hes_id)
    cd30_out = os.path.join(cd30_dir, cd30_id)
    os.makedirs(hes_out, exist_ok=True)
    os.makedirs(cd30_out, exist_ok=True)
    
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
    
    # Étape 2: Alignement des images basse résolution
    print("\n[2/4] Alignement des images...")
    transformation, aligned_cd30_lr, success = align_images_orb(lowres_hes_np, lowres_cd30_np)
    
    if not success:
        print("  ⚠ Utilisation des images non alignées (alignement échoué)")
        transformation = None
    
    # Étape 3: Calcul du masque de tissu
    print("\n[3/4] Calcul du masque de tissu...")
    mask_hes = compute_tissue_mask(lowres_hes)
    
    # Étape 4: Extraction des patches alignés
    print("\n[4/4] Extraction des patches...")
    w0_hes, h0_hes = slide_hes.level_dimensions[0]
    downsample_hes = int(slide_hes.level_downsamples[lowres_level])
    downsample_cd30 = int(slide_cd30.level_downsamples[lowres_level])
    
    patch_count = 0
    
    for y in range(0, h0_hes, patch_size):
        for x in range(0, w0_hes, patch_size):
            
            # Vérifier la présence de tissu
            if not patch_has_tissue(x, y, mask_hes, downsample_hes):
                continue
            
            # Extraire le patch HES
            patch_hes = slide_hes.read_region(
                (x, y), 0, (patch_size, patch_size)
            ).convert("RGB")
            
            # Extraire le patch CD30 correspondant (avec transformation si disponible)
            if transformation is not None:
                # Appliquer la transformation inverse aux coordonnées
                scale_factor = slide_hes.level_downsamples[lowres_level]
                
                # Coordonnées au niveau lowres
                x_lr = x / scale_factor
                y_lr = y / scale_factor
                
                # Inverser la transformation
                try:
                    transform_matrix = transformation.params[:2]
                    # Créer une matrice 3x3 pour l'inversion
                    full_matrix = np.vstack([transform_matrix, [0, 0, 1]])
                    inv_matrix = np.linalg.inv(full_matrix)
                    
                    # Appliquer l'inverse aux coordonnées lowres
                    point = np.array([x_lr, y_lr, 1])
                    transformed_point = inv_matrix @ point
                    
                    x_cd30_lr = transformed_point[0]
                    y_cd30_lr = transformed_point[1]
                    
                    # Revenir au niveau 0
                    x_cd30 = int(x_cd30_lr * scale_factor)
                    y_cd30 = int(y_cd30_lr * scale_factor)
                    
                except Exception as e:
                    print(f"  ⚠ Erreur de transformation: {e}, utilisation coordonnées directes")
                    x_cd30, y_cd30 = x, y
            else:
                x_cd30, y_cd30 = x, y
            
            # Vérifier que les coordonnées sont dans les limites
            w0_cd30, h0_cd30 = slide_cd30.level_dimensions[0]
            if x_cd30 < 0 or y_cd30 < 0 or x_cd30 + patch_size > w0_cd30 or y_cd30 + patch_size > h0_cd30:
                continue
            
            # Extraire le patch CD30
            patch_cd30 = slide_cd30.read_region(
                (x_cd30, y_cd30), 0, (patch_size, patch_size)
            ).convert("RGB")
            
            # Sauvegarder les patches
            name_hes = f"{hes_id}_x{x}_y{y}.png"
            name_cd30 = f"{cd30_id}_x{x}_y{y}.png"  # Utilise les mêmes coordonnées pour l'appariement
            
            patch_hes.save(os.path.join(hes_out, name_hes))
            patch_cd30.save(os.path.join(cd30_out, name_cd30))
            
            patch_count += 1
    
    print(f"\n✓ {patch_count} paires de patches extraites")
    
    # Fermer les slides
    slide_hes.close()
    slide_cd30.close()
    
    return patch_count


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
for idx, (base_id, paths) in enumerate(pairs.items(), 1):
    print(f"\n{'*'*80}")
    print(f"Paire {idx}/{len(pairs)}: {base_id}")
    print(f"{'*'*80}")
    
    try:
        patch_count = process_slide_pair(paths['hes'], paths['cd30'], hes_dir, cd30_dir)
        total_patches += patch_count
    except Exception as e:
        print(f"\n✗ Erreur lors du traitement de {base_id}: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n{'='*80}")
print("TRAITEMENT TERMINÉ")
print(f"{'='*80}")
print(f"Total: {total_patches} paires de patches extraites")
print(f"Patches HES sauvegardés dans: {hes_dir}")
print(f"Patches CD30 sauvegardés dans: {cd30_dir}")
