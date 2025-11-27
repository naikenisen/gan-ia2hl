import os
import openslide
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings
warnings.filterwarnings('ignore')

# Configuration
input_folder = "/gold/data_feasibility"
output_file = "alignment_overview.jpg"

# Liste des patients à traiter
patient_ids = ["AHL002"]

patch_size = 2000
region_size = 12000
stride_region = 12000
lowres_level = 2
tissue_threshold = 0.60


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

def create_overlay(hes_img, cd30_img, alpha=0.5):
    """
    Crée une image overlay pour visualiser l'alignement.
    H&E en rouge, CD30 en cyan.
    """
    overlay = np.zeros_like(hes_img)
    overlay[:, :, 0] = (hes_img[:, :, 0] * alpha).astype(np.uint8)  # Rouge pour H&E
    overlay[:, :, 1] = (cd30_img[:, :, 1] * alpha).astype(np.uint8)  # Cyan pour CD30
    overlay[:, :, 2] = (cd30_img[:, :, 2] * alpha).astype(np.uint8)
    return overlay

def register_images_sift_ransac(fixed_img_np, moving_img_np):
    """
    Recale l'image moving sur l'image fixed avec SIFT + RANSAC.
    Méthode robuste basée sur la détection de points caractéristiques.
    
    Args:
        fixed_img_np: Image fixe (H&E) en numpy array RGB
        moving_img_np: Image à recaler (CD30) en numpy array RGB
    
    Returns:
        registered_img_np: Image CD30 recalée en numpy array RGB
        transform_matrix: Matrice de transformation affine 2x3
    """
    print("  Registration SIFT + RANSAC en cours...")
    
    # Convertir en niveaux de gris pour la détection de features
    fixed_gray = cv2.cvtColor(fixed_img_np, cv2.COLOR_RGB2GRAY)
    moving_gray = cv2.cvtColor(moving_img_np, cv2.COLOR_RGB2GRAY)
    
    # Initialiser le détecteur SIFT
    sift = cv2.SIFT_create(nfeatures=5000)
    
    # Détecter les keypoints et calculer les descripteurs
    print("    Détection des keypoints SIFT...")
    kp_fixed, desc_fixed = sift.detectAndCompute(fixed_gray, None)
    kp_moving, desc_moving = sift.detectAndCompute(moving_gray, None)
    
    print(f"    H&E: {len(kp_fixed)} keypoints, CD30: {len(kp_moving)} keypoints")
    
    if len(kp_fixed) < 10 or len(kp_moving) < 10:
        print("    ⚠ Pas assez de keypoints détectés, retour image originale")
        return moving_img_np, None
    
    # Matcher les descripteurs avec FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    print("    Matching des descripteurs...")
    matches = flann.knnMatch(desc_moving, desc_fixed, k=2)
    
    # Filtrer les bons matches avec le ratio test de Lowe
    good_matches = []
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    print(f"    {len(good_matches)} bons matches trouvés")
    
    if len(good_matches) < 10:
        print("    ⚠ Pas assez de bons matches, retour image originale")
        return moving_img_np, None
    
    # Extraire les coordonnées des points matchés
    src_pts = np.float32([kp_moving[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_fixed[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Estimer la transformation affine avec RANSAC
    print("    Estimation de la transformation avec RANSAC...")
    transform_matrix, inliers = cv2.estimateAffinePartial2D(
        src_pts, 
        dst_pts, 
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=5000,
        confidence=0.999
    )
    
    if transform_matrix is None:
        print("    ⚠ Échec de l'estimation RANSAC, retour image originale")
        return moving_img_np, None
    
    n_inliers = np.sum(inliers)
    print(f"    Transformation estimée avec {n_inliers}/{len(good_matches)} inliers")
    
    # Appliquer la transformation à l'image couleur
    h, w = fixed_img_np.shape[:2]
    registered_img_np = cv2.warpAffine(
        moving_img_np,
        transform_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )
    
    print("  ✓ Registration terminée")
    
    return registered_img_np, transform_matrix

def process_patient(patient_id):
    """Traite un patient et retourne les images annotées."""
    print(f"\nTraitement de {patient_id}...")
    
    hes_file = f"{patient_id}_HES.svs"
    cd30_file = f"{patient_id}_CD30.svs"
    hes_path = os.path.join(input_folder, hes_file)
    cd30_path = os.path.join(input_folder, cd30_file)
    
    # Charger les lames
    slide_hes = openslide.OpenSlide(hes_path)
    slide_cd30 = openslide.OpenSlide(cd30_path)
    
    # Charger images basse résolution
    w_lr_hes, h_lr_hes = slide_hes.level_dimensions[lowres_level]
    w_lr_cd30, h_lr_cd30 = slide_cd30.level_dimensions[lowres_level]
    
    lowres_hes = slide_hes.read_region((0, 0), lowres_level, (w_lr_hes, h_lr_hes)).convert("RGB")
    lowres_cd30 = slide_cd30.read_region((0, 0), lowres_level, (w_lr_cd30, h_lr_cd30)).convert("RGB")
    
    lowres_hes_np = np.array(lowres_hes)
    lowres_cd30_np = np.array(lowres_cd30)
    
    # ÉTAPE DE REGISTRATION: Recaler CD30 sur H&E avec SIFT + RANSAC
    print("  Démarrage de la registration SIFT + RANSAC...")
    lowres_cd30_registered, transform_params = register_images_sift_ransac(lowres_hes_np, lowres_cd30_np)
    print("  CD30 recalé sur H&E")
    
    # Calculer le masque de tissu
    mask_hes = compute_tissue_mask(lowres_hes)
    
    # Dimensions
    w0_hes, h0_hes = slide_hes.level_dimensions[0]
    w0_cd30, h0_cd30 = slide_cd30.level_dimensions[0]
    downsample_hes = int(slide_hes.level_downsamples[lowres_level])
    downsample_cd30 = int(slide_cd30.level_downsamples[lowres_level])
    
    # Identifier les régions valides
    valid_regions = []
    region_index = 0
    
    for region_y in range(0, h0_hes, stride_region):
        for region_x in range(0, w0_hes, stride_region):
            if region_x + region_size > w0_hes or region_y + region_size > h0_hes:
                continue
            
            region_x_lr = int(region_x / downsample_hes)
            region_y_lr = int(region_y / downsample_hes)
            region_size_lr = int(region_size / downsample_hes)
            
            region_mask = mask_hes[region_y_lr:region_y_lr+region_size_lr, region_x_lr:region_x_lr+region_size_lr]
            if region_mask.size == 0 or np.mean(region_mask > 0) < tissue_threshold:
                continue
            
            region_x_cd30_lr = int(region_x / downsample_cd30)
            region_y_cd30_lr = int(region_y / downsample_cd30)
            region_size_cd30_lr = int(region_size / downsample_cd30)
            
            if region_x_cd30_lr + region_size_cd30_lr > w_lr_cd30 or region_y_cd30_lr + region_size_cd30_lr > h_lr_cd30:
                continue
            
            valid_regions.append({
                'index': region_index,
                'x': region_x,
                'y': region_y,
                'x_lr': region_x_lr,
                'y_lr': region_y_lr,
                'size_lr': region_size_lr
            })
            region_index += 1
    
    print(f"  Trouvé {len(valid_regions)} régions valides")
    
    # Créer les images annotées (AVEC et SANS registration)
    img_hes_annotated = lowres_hes_np.copy()
    img_cd30_no_reg_annotated = lowres_cd30_np.copy()  # SANS registration
    img_cd30_registered_annotated = lowres_cd30_registered.copy()  # AVEC registration
    
    # Annoter les régions
    for region in valid_regions:
        # HES
        cv2.rectangle(
            img_hes_annotated,
            (region['x_lr'], region['y_lr']),
            (region['x_lr'] + region['size_lr'], region['y_lr'] + region['size_lr']),
            (255, 0, 0), 8
        )
        
        # CD30 - Annoter les deux versions
        region_x_cd30_lr = int(region['x'] / downsample_cd30)
        region_y_cd30_lr = int(region['y'] / downsample_cd30)
        region_size_cd30_lr = int(region_size / downsample_cd30)
        
        # CD30 SANS registration
        cv2.rectangle(
            img_cd30_no_reg_annotated,
            (region_x_cd30_lr, region_y_cd30_lr),
            (region_x_cd30_lr + region_size_cd30_lr, region_y_cd30_lr + region_size_cd30_lr),
            (0, 0, 255), 8
        )
        
        # CD30 AVEC registration
        cv2.rectangle(
            img_cd30_registered_annotated,
            (region_x_cd30_lr, region_y_cd30_lr),
            (region_x_cd30_lr + region_size_cd30_lr, region_y_cd30_lr + region_size_cd30_lr),
            (0, 255, 0), 8  # Vert pour différencier
        )
    
    # Créer des overlays pour visualiser l'amélioration
    overlay_no_reg = create_overlay(lowres_hes_np, lowres_cd30_np)
    overlay_registered = create_overlay(lowres_hes_np, lowres_cd30_registered)
    
    # Nettoyer
    slide_hes.close()
    slide_cd30.close()
    
    return (img_hes_annotated, img_cd30_no_reg_annotated, img_cd30_registered_annotated, 
            overlay_no_reg, overlay_registered, len(valid_regions))


# Traiter tous les patients
print("="*80)
print("GÉNÉRATION DE LA FIGURE GLOBALE D'ALIGNEMENT")
print("="*80)

all_images = []
for patient_id in patient_ids:
    try:
        img_hes, img_cd30_no_reg, img_cd30_reg, overlay_no_reg, overlay_reg, n_regions = process_patient(patient_id)
        all_images.append({
            'patient_id': patient_id,
            'hes': img_hes,
            'cd30_no_reg': img_cd30_no_reg,
            'cd30_registered': img_cd30_reg,
            'overlay_no_reg': overlay_no_reg,
            'overlay_registered': overlay_reg,
            'n_regions': n_regions
        })
    except Exception as e:
        print(f"  ⚠ Erreur pour {patient_id}: {e}")
        import traceback
        traceback.print_exc()
        continue

# Vérifier qu'au moins un patient a été traité avec succès
if len(all_images) == 0:
    print("\n" + "="*80)
    print("ERREUR: Aucun patient n'a pu être traité avec succès!")
    print("="*80)
    exit(1)

# Créer la grande figure - 5 colonnes pour comparaison complète
print("\n" + "="*80)
print("CRÉATION DE LA FIGURE FINALE")
print("="*80)

n_patients = len(all_images)
fig, axes = plt.subplots(n_patients, 5, figsize=(30, 5*n_patients))

# S'assurer que axes est un tableau 2D même avec un seul patient
if n_patients == 1:
    axes = axes.reshape(1, -1)

for idx, data in enumerate(all_images):
    # Colonne 1: H&E
    axes[idx, 0].imshow(data['hes'])
    axes[idx, 0].set_title(f"{data['patient_id']} - H&E\n{data['n_regions']} régions", 
                           fontsize=12, fontweight='bold')
    axes[idx, 0].axis('off')
    
    # Colonne 2: CD30 SANS registration (rouge)
    axes[idx, 1].imshow(data['cd30_no_reg'])
    axes[idx, 1].set_title(f"CD30 SANS registration\nBoîtes rouges", 
                           fontsize=12, fontweight='bold', color='red')
    axes[idx, 1].axis('off')
    
    # Colonne 3: Overlay SANS registration
    axes[idx, 2].imshow(data['overlay_no_reg'])
    axes[idx, 2].set_title(f"Overlay SANS registration\n(H&E=rouge, CD30=cyan)", 
                           fontsize=12, fontweight='bold', color='orange')
    axes[idx, 2].axis('off')
    
    # Colonne 4: CD30 AVEC registration (vert)
    axes[idx, 3].imshow(data['cd30_registered'])
    axes[idx, 3].set_title(f"CD30 AVEC registration Elastix\nBoîtes vertes", 
                           fontsize=12, fontweight='bold', color='green')
    axes[idx, 3].axis('off')
    
    # Colonne 5: Overlay AVEC registration
    axes[idx, 4].imshow(data['overlay_registered'])
    axes[idx, 4].set_title(f"Overlay AVEC registration\n(H&E=rouge, CD30=cyan)", 
                           fontsize=12, fontweight='bold', color='darkgreen')
    axes[idx, 4].axis('off')

plt.suptitle('Comparaison: Alignement SANS vs AVEC Registration Elastix\nLes overlays montrent la qualité de l\'alignement', 
             fontsize=18, fontweight='bold', y=0.998)
plt.tight_layout()

# Sauvegarder en haute qualité
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='jpg')
print(f"\n✓ Figure sauvegardée: {output_file}")
print(f"  Format: JPEG haute qualité (300 DPI)")
print(f"  Patients inclus: {', '.join([d['patient_id'] for d in all_images])}")
print(f"  Total régions: {sum([d['n_regions'] for d in all_images])}")
print("\n" + "="*80)
print("STRUCTURE DE LA FIGURE (5 colonnes)")
print("="*80)
print("  Colonne 1: H&E - Image de référence avec boîtes bleues")
print("  Colonne 2: CD30 SANS registration - Boîtes ROUGES (alignement initial)")
print("  Colonne 3: OVERLAY SANS registration - Visualisation de l'alignement initial")
print("              (H&E=rouge, CD30=cyan, si bien aligné → blanc)")
print("  Colonne 4: CD30 AVEC registration Elastix - Boîtes VERTES (alignement amélioré)")
print("  Colonne 5: OVERLAY AVEC registration - Visualisation de l'alignement final")
print("              (H&E=rouge, CD30=cyan, si bien aligné → blanc)")
print("="*80)
print("\n✓ Visualisation terminée!")