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
warnings.filterwarnings('ignore')

# Configuration
input_folder = "/gold/data_feasibility"
output_folder = "./visualizations_ORB"
os.makedirs(output_folder, exist_ok=True)

patch_size = 2000
region_size = 12000
lowres_level = 2

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


def visualize_orb_detection_and_matching(img_fixed, img_moving, patient_id, region_idx):
    """
    Visualise la détection ORB et l'appariement des points.
    Retourne aussi la transformation calculée.
    """
    # Conversion en niveaux de gris
    gray_fixed = cv2.cvtColor(img_fixed, cv2.COLOR_RGB2GRAY)
    gray_moving = cv2.cvtColor(img_moving, cv2.COLOR_RGB2GRAY)
    
    # Détection ORB
    orb = cv2.ORB_create(nfeatures=5000)
    kp1, desc1 = orb.detectAndCompute(gray_fixed, None)
    kp2, desc2 = orb.detectAndCompute(gray_moving, None)
    
    if desc1 is None or desc2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("Pas assez de points détectés")
        return None, None, None, False
    
    # Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    num_good_matches = min(len(matches), max(50, int(len(matches) * 0.15)))
    good_matches = matches[:num_good_matches]
    
    if len(good_matches) < 4:
        print("Pas assez de correspondances")
        return None, None, None, False
    
    # Extraire les points
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    
    # RANSAC
    try:
        model, inliers = ransac(
            (src_pts, dst_pts),
            AffineTransform,
            min_samples=3,
            residual_threshold=5.0,
            max_trials=1000
        )
        
        if model is None or np.sum(inliers) < 4:
            print("RANSAC échoué")
            return None, None, None, False
        
        num_inliers = np.sum(inliers)
        print(f"✓ {num_inliers} inliers sur {len(good_matches)} matches")
        
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
        
        # === FIGURE 1: Détection des points ORB ===
        fig1, axes = plt.subplots(1, 3, figsize=(20, 7))
        
        # Points détectés dans HES
        img1_kp = cv2.drawKeypoints(img_fixed, kp1, None, 
                                     color=(0, 255, 0), 
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        axes[0].imshow(img1_kp)
        axes[0].set_title(f'HES (référence)\n{len(kp1)} points ORB détectés', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Points détectés dans CD30
        img2_kp = cv2.drawKeypoints(img_moving, kp2, None, 
                                     color=(255, 0, 0), 
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        axes[1].imshow(img2_kp)
        axes[1].set_title(f'CD30 (à aligner)\n{len(kp2)} points ORB détectés', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Correspondances avec inliers/outliers
        img_matches = np.hstack([img_fixed, img_moving])
        axes[2].imshow(img_matches)
        
        offset_x = img_fixed.shape[1]
        
        # Dessiner les matches (outliers en rouge, inliers en vert)
        for idx, match in enumerate(good_matches):
            pt1 = tuple(map(int, kp1[match.queryIdx].pt))
            pt2 = tuple(map(int, kp2[match.trainIdx].pt))
            pt2_shifted = (pt2[0] + offset_x, pt2[1])
            
            if inliers[idx]:
                color = 'lime'
                alpha = 0.8
                linewidth = 1.5
            else:
                color = 'red'
                alpha = 0.3
                linewidth = 0.8
            
            axes[2].plot([pt1[0], pt2_shifted[0]], [pt1[1], pt2_shifted[1]], 
                        color=color, alpha=alpha, linewidth=linewidth)
            axes[2].plot(pt1[0], pt1[1], 'o', color='cyan', markersize=3)
            axes[2].plot(pt2_shifted[0], pt2_shifted[1], 'o', color='yellow', markersize=3)
        
        axes[2].set_title(f'Appariement des points\n{num_inliers} inliers (vert) / {len(good_matches)-num_inliers} outliers (rouge)', 
                         fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(f'Patient {patient_id} - Région {region_idx}\nÉtape 1: Détection ORB et appariement', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        output_path1 = os.path.join(output_folder, f'{patient_id}_region{region_idx}_1_ORB_detection.png')
        plt.savefig(output_path1, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  → Sauvegardé: {output_path1}")
        
        return model, aligned_img, inliers, True
        
    except Exception as e:
        print(f"Erreur: {e}")
        return None, None, None, False


def visualize_transformation_effect(img_fixed, img_moving, aligned_img, patient_id, region_idx):
    """Visualise l'effet de la transformation: avant/après alignement."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Ligne 1: Images complètes
    axes[0, 0].imshow(img_fixed)
    axes[0, 0].set_title('HES (référence)', fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_moving)
    axes[0, 1].set_title('CD30 (avant alignement)', fontsize=13, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(aligned_img)
    axes[0, 2].set_title('CD30 (après alignement)', fontsize=13, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Ligne 2: Différences et overlay
    # Différence avant alignement
    diff_before = np.abs(img_fixed.astype(float) - img_moving.astype(float)).astype(np.uint8)
    diff_before_gray = cv2.cvtColor(diff_before, cv2.COLOR_RGB2GRAY)
    axes[1, 0].imshow(diff_before_gray, cmap='hot')
    axes[1, 0].set_title('Différence AVANT alignement', fontsize=13, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Différence après alignement
    diff_after = np.abs(img_fixed.astype(float) - aligned_img.astype(float)).astype(np.uint8)
    diff_after_gray = cv2.cvtColor(diff_after, cv2.COLOR_RGB2GRAY)
    axes[1, 1].imshow(diff_after_gray, cmap='hot')
    axes[1, 1].set_title('Différence APRÈS alignement', fontsize=13, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Overlay (checkerboard)
    checkerboard = create_checkerboard_overlay(img_fixed, aligned_img, square_size=200)
    axes[1, 2].imshow(checkerboard)
    axes[1, 2].set_title('Overlay en damier (HES/CD30 aligné)', fontsize=13, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Patient {patient_id} - Région {region_idx}\nÉtape 2: Effet de la transformation affine', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = os.path.join(output_folder, f'{patient_id}_region{region_idx}_2_transformation_effect.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Sauvegardé: {output_path}")


def create_checkerboard_overlay(img1, img2, square_size=100):
    """Crée un overlay en damier de deux images."""
    h, w = img1.shape[:2]
    result = img1.copy()
    
    for i in range(0, h, square_size):
        for j in range(0, w, square_size):
            if (i // square_size + j // square_size) % 2 == 0:
                i_end = min(i + square_size, h)
                j_end = min(j + square_size, w)
                result[i:i_end, j:j_end] = img2[i:i_end, j:j_end]
    
    return result


def visualize_patch_extraction(slide_hes, slide_cd30, region_x, region_y, 
                               transformation, downsample_hes, downsample_cd30,
                               patient_id, region_idx):
    """Visualise l'impact de la transformation sur un patch spécifique."""
    
    # Sélectionner un patch au milieu de la région
    patch_x = region_x + region_size // 2
    patch_y = region_y + region_size // 2
    
    # Extraire le patch HES
    patch_hes = slide_hes.read_region((patch_x, patch_y), 0, (patch_size, patch_size)).convert("RGB")
    patch_hes_np = np.array(patch_hes)
    
    # Calculer la position CD30 SANS transformation (naïve)
    patch_cd30_naive = slide_cd30.read_region((patch_x, patch_y), 0, (patch_size, patch_size)).convert("RGB")
    patch_cd30_naive_np = np.array(patch_cd30_naive)
    
    # Calculer la position CD30 AVEC transformation
    x_local = patch_x - region_x
    y_local = patch_y - region_y
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
        
        patch_cd30_aligned = slide_cd30.read_region((x_cd30, y_cd30), 0, (patch_size, patch_size)).convert("RGB")
        patch_cd30_aligned_np = np.array(patch_cd30_aligned)
        
    except Exception as e:
        print(f"Erreur transformation patch: {e}")
        x_cd30, y_cd30 = patch_x, patch_y
        patch_cd30_aligned_np = patch_cd30_naive_np
    
    # Créer la figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Ligne 1: Les trois patches
    axes[0, 0].imshow(patch_hes_np)
    axes[0, 0].set_title(f'Patch HES (référence)\nPosition: ({patch_x}, {patch_y})', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(patch_cd30_naive_np)
    axes[0, 1].set_title(f'Patch CD30 SANS transformation\nPosition: ({patch_x}, {patch_y})\n(même coordonnées que HES)', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(patch_cd30_aligned_np)
    axes[0, 2].set_title(f'Patch CD30 AVEC transformation\nPosition: ({x_cd30}, {y_cd30})\n(corrigée par ORB+RANSAC)', 
                        fontsize=12, fontweight='bold', color='green')
    axes[0, 2].axis('off')
    
    # Ligne 2: Comparaisons
    # Overlay HES vs CD30 sans transformation
    overlay1 = cv2.addWeighted(patch_hes_np, 0.5, patch_cd30_naive_np, 0.5, 0)
    axes[1, 0].imshow(overlay1)
    axes[1, 0].set_title('Overlay: HES + CD30 non aligné\n(désalignement visible)', 
                        fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Overlay HES vs CD30 avec transformation
    overlay2 = cv2.addWeighted(patch_hes_np, 0.5, patch_cd30_aligned_np, 0.5, 0)
    axes[1, 1].imshow(overlay2)
    axes[1, 1].set_title('Overlay: HES + CD30 aligné\n(meilleur alignement)', 
                        fontsize=12, fontweight='bold', color='green')
    axes[1, 1].axis('off')
    
    # Différence entre les deux approches
    diff = np.abs(patch_cd30_naive_np.astype(float) - patch_cd30_aligned_np.astype(float)).astype(np.uint8)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    axes[1, 2].imshow(diff_gray, cmap='hot')
    axes[1, 2].set_title(f'Différence entre CD30 non aligné vs aligné\nDécalage en pixels: Δx={x_cd30-patch_x}, Δy={y_cd30-patch_y}', 
                        fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Patient {patient_id} - Région {region_idx}\nÉtape 3: Impact de la transformation sur l\'extraction de patches ({patch_size}×{patch_size}px)', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    output_path = os.path.join(output_folder, f'{patient_id}_region{region_idx}_3_patch_impact.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Sauvegardé: {output_path}")


def process_one_slide_pair_visualization(hes_path, cd30_path):
    """Traite une paire de slides et génère toutes les visualisations."""
    
    print(f"\n{'='*80}")
    print(f"Traitement de: {os.path.basename(hes_path)}")
    print(f"{'='*80}")
    
    slide_hes = openslide.OpenSlide(hes_path)
    slide_cd30 = openslide.OpenSlide(cd30_path)
    
    patient_id = os.path.splitext(os.path.basename(hes_path))[0].replace("_HES", "")
    
    # Charger les images basse résolution
    print("\n[1/3] Chargement des images basse résolution...")
    w_lr_hes, h_lr_hes = slide_hes.level_dimensions[lowres_level]
    w_lr_cd30, h_lr_cd30 = slide_cd30.level_dimensions[lowres_level]
    
    lowres_hes = slide_hes.read_region((0, 0), lowres_level, (w_lr_hes, h_lr_hes)).convert("RGB")
    lowres_cd30 = slide_cd30.read_region((0, 0), lowres_level, (w_lr_cd30, h_lr_cd30)).convert("RGB")
    
    lowres_hes_np = np.array(lowres_hes)
    lowres_cd30_np = np.array(lowres_cd30)
    
    # Calculer le masque de tissu
    print("\n[2/3] Calcul du masque de tissu...")
    mask_hes = compute_tissue_mask(lowres_hes)
    
    # Trouver une région avec du tissu
    print("\n[3/3] Recherche d'une région avec tissu...")
    w0_hes, h0_hes = slide_hes.level_dimensions[0]
    downsample_hes = int(slide_hes.level_downsamples[lowres_level])
    downsample_cd30 = int(slide_cd30.level_downsamples[lowres_level])
    
    region_found = False
    region_idx = 0
    
    for region_y in range(0, h0_hes, region_size):
        for region_x in range(0, w0_hes, region_size):
            if region_x + region_size > w0_hes or region_y + region_size > h0_hes:
                continue
            
            region_x_lr = int(region_x / downsample_hes)
            region_y_lr = int(region_y / downsample_hes)
            region_size_lr = int(region_size / downsample_hes)
            
            region_mask = mask_hes[region_y_lr:region_y_lr+region_size_lr, 
                                   region_x_lr:region_x_lr+region_size_lr]
            
            if region_mask.size == 0 or np.mean(region_mask > 0) < 0.60:
                continue
            
            print(f"\n✓ Région trouvée à x={region_x}, y={region_y}")
            
            # Extraire les sous-régions
            region_hes_lr = lowres_hes_np[region_y_lr:region_y_lr+region_size_lr, 
                                          region_x_lr:region_x_lr+region_size_lr]
            
            region_x_cd30_lr = int(region_x / downsample_cd30)
            region_y_cd30_lr = int(region_y / downsample_cd30)
            region_size_cd30_lr = int(region_size / downsample_cd30)
            
            if region_x_cd30_lr + region_size_cd30_lr > w_lr_cd30 or region_y_cd30_lr + region_size_cd30_lr > h_lr_cd30:
                continue
            
            region_cd30_lr = lowres_cd30_np[region_y_cd30_lr:region_y_cd30_lr+region_size_cd30_lr, 
                                           region_x_cd30_lr:region_x_cd30_lr+region_size_cd30_lr]
            
            print("\n>>> Génération de la visualisation 1: Détection ORB et appariement")
            transformation, aligned_img, inliers, success = visualize_orb_detection_and_matching(
                region_hes_lr, region_cd30_lr, patient_id, region_idx
            )
            
            if not success:
                print("Alignement échoué, recherche d'une autre région...")
                continue
            
            print("\n>>> Génération de la visualisation 2: Effet de la transformation")
            visualize_transformation_effect(
                region_hes_lr, region_cd30_lr, aligned_img, patient_id, region_idx
            )
            
            print("\n>>> Génération de la visualisation 3: Impact sur les patches")
            visualize_patch_extraction(
                slide_hes, slide_cd30, region_x, region_y,
                transformation, downsample_hes, downsample_cd30,
                patient_id, region_idx
            )
            
            region_found = True
            break
        
        if region_found:
            break
    
    slide_hes.close()
    slide_cd30.close()
    
    if region_found:
        print(f"\n{'='*80}")
        print(f"✓ Visualisations générées avec succès pour {patient_id}")
        print(f"{'='*80}")
    else:
        print(f"\n✗ Aucune région appropriée trouvée")


# ============================================================================
# EXÉCUTION PRINCIPALE
# ============================================================================

print(f"\n{'='*80}")
print("VISUALISATION DU PROCESSUS ORB + RANSAC")
print(f"{'='*80}")
print(f"Dossier d'entrée: {input_folder}")
print(f"Dossier de sortie: {output_folder}")

# Trouver les fichiers
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

print(f"\n{len(pairs)} paires disponibles")

# Traiter une paire spécifique
if pairs:
    # Choisir le patient AHL006
    target_patient = "AHL006"
    
    if target_patient in pairs:
        patient_id = target_patient
        paths = pairs[target_patient]
        print(f"\nTraitement de la paire: {patient_id}")
        process_one_slide_pair_visualization(paths['hes'], paths['cd30'])
    else:
        print(f"\n✗ Patient {target_patient} non trouvé dans les paires disponibles")
        print(f"Paires disponibles: {list(pairs.keys())}")
    
    print(f"\n{'='*80}")
    print(f"✓ TERMINÉ")
    print(f"Les visualisations sont sauvegardées dans: {output_folder}")
    print(f"{'='*80}")
else:
    print("\n✗ Aucune paire de slides trouvée")
