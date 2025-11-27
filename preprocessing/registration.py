import os
import openslide
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.feature import ORB
from skimage.measure import ransac
from skimage.transform import AffineTransform
import SimpleITK as sitk
import warnings
warnings.filterwarnings('ignore')

# Configuration
input_folder = "/gold/data_feasibility"
output_file = "registration.jpg"

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

def apply_otsu_segmentation(img_rgb):
    """
    Applique la segmentation Otsu pour extraire uniquement les tissus.
    Les zones non-tissulaires sont mises en blanc.
    """
    mask = compute_tissue_mask(img_rgb)
    img_np = np.array(img_rgb)
    img_segmented = img_np.copy()
    img_segmented[mask == 0] = 255  # Mettre le background en blanc
    return img_segmented, mask

def extract_contours(mask, thickness=3):
    """
    Extrait les contours d'un masque binaire.
    
    Args:
        mask: Masque binaire (0 ou 255)
        thickness: Épaisseur des contours en pixels
    
    Returns:
        contour_img: Image RGB avec les contours en noir sur fond blanc
    """
    # Détecter les contours avec Canny
    edges = cv2.Canny(mask, 50, 150)
    
    # Dilater légèrement pour avoir des contours plus visibles
    kernel = np.ones((thickness, thickness), np.uint8)
    edges_thick = cv2.dilate(edges, kernel, iterations=1)
    
    # Créer une image RGB: contours en noir, fond en blanc
    contour_img = np.ones((mask.shape[0], mask.shape[1], 3), dtype=np.uint8) * 255
    contour_img[edges_thick > 0] = [0, 0, 0]  # Noir pour les contours
    
    return contour_img, edges_thick

def create_contour_overlay(hes_contours, cd30_contours):
    """
    Crée un overlay des contours: H&E en rouge, CD30 en vert.
    Zones bien alignées = jaune.
    """
    overlay = np.ones_like(hes_contours) * 255
    
    # H&E en rouge
    hes_mask = np.any(hes_contours < 200, axis=2)
    overlay[hes_mask] = [255, 0, 0]
    
    # CD30 en vert
    cd30_mask = np.any(cd30_contours < 200, axis=2)
    overlay[cd30_mask] = [0, 255, 0]
    
    # Intersection en jaune
    intersection = hes_mask & cd30_mask
    overlay[intersection] = [255, 255, 0]
    
    return overlay

def create_overlay(hes_img, cd30_img, alpha=0.6):
    """
    Crée une image overlay pour visualiser l'alignement.
    H&E en magenta, CD30 en vert. Zones bien alignées = jaune/blanc.
    """
    # S'assurer que les deux images ont la même taille
    if hes_img.shape != cd30_img.shape:
        # Redimensionner cd30 pour correspondre à hes
        cd30_img = cv2.resize(cd30_img, (hes_img.shape[1], hes_img.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Convertir en niveaux de gris pour meilleur contraste
    hes_gray = cv2.cvtColor(hes_img, cv2.COLOR_RGB2GRAY)
    cd30_gray = cv2.cvtColor(cd30_img, cv2.COLOR_RGB2GRAY)
    
    # Créer overlay avec couleurs complémentaires
    overlay = np.zeros_like(hes_img)
    overlay[:, :, 0] = np.clip(hes_gray * alpha + cd30_gray * (1-alpha), 0, 255).astype(np.uint8)  # Rouge (Magenta + Vert)
    overlay[:, :, 1] = np.clip(cd30_gray * 1.2, 0, 255).astype(np.uint8)  # Vert fort pour CD30
    overlay[:, :, 2] = np.clip(hes_gray * alpha, 0, 255).astype(np.uint8)  # Bleu (Magenta)
    
    return overlay

def create_checkerboard(hes_img, cd30_img, square_size=200):
    """
    Crée une visualisation en damier pour comparer avant/après registration.
    """
    # S'assurer que les deux images ont la même taille
    if hes_img.shape != cd30_img.shape:
        cd30_img = cv2.resize(cd30_img, (hes_img.shape[1], hes_img.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    h, w = hes_img.shape[:2]
    checkerboard = hes_img.copy()
    
    # Créer le pattern de damier
    for y in range(0, h, square_size):
        for x in range(0, w, square_size):
            # Alterner entre H&E et CD30
            if ((y // square_size) + (x // square_size)) % 2 == 0:
                y_end = min(y + square_size, h)
                x_end = min(x + square_size, w)
                checkerboard[y:y_end, x:x_end] = cd30_img[y:y_end, x:x_end]
    
    return checkerboard

def create_difference_map(hes_img, cd30_img):
    """
    Crée une carte de différence pour visualiser le désalignement.
    Plus c'est rouge, plus il y a de différence (mauvais alignement).
    """
    # S'assurer que les deux images ont la même taille
    if hes_img.shape != cd30_img.shape:
        cd30_img = cv2.resize(cd30_img, (hes_img.shape[1], hes_img.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # Convertir en niveaux de gris
    hes_gray = cv2.cvtColor(hes_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    cd30_gray = cv2.cvtColor(cd30_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Calculer la différence absolue
    diff = np.abs(hes_gray - cd30_gray)
    
    # Normaliser et appliquer une colormap "hot" (bleu=bon, rouge=mauvais)
    diff_norm = (diff / diff.max() * 255).astype(np.uint8)
    diff_colored = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
    diff_colored = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)
    
    return diff_colored

def register_images_elastix(fixed_img_np, moving_img_np, fixed_mask=None, moving_mask=None, use_contours=True):
    """
    Recale l'image moving sur l'image fixed avec SimpleITK.
    Utilise ImageRegistrationMethod avec transformation affine.
    
    Args:
        fixed_img_np: Image fixe (H&E segmentée) en numpy array RGB
        moving_img_np: Image à recaler (CD30 segmentée) en numpy array RGB
        fixed_mask: Masque optionnel pour l'image fixe
        moving_mask: Masque optionnel pour l'image moving
        use_contours: Si True, effectue la registration sur les contours plutôt que l'intérieur
    
    Returns:
        registered_img_np: Image CD30 recalée en numpy array RGB
        transform: Transformation appliquée
        fixed_contours: Contours de l'image fixe (si use_contours=True)
        moving_contours_registered: Contours recalés (si use_contours=True)
    """
    if use_contours:
        print("  Registration SimpleITK en cours (sur les CONTOURS)...")
        
        # Extraire les contours des masques
        if fixed_mask is None:
            fixed_mask = compute_tissue_mask(fixed_img_np)
        if moving_mask is None:
            moving_mask = compute_tissue_mask(moving_img_np)
        
        fixed_contours, fixed_edges = extract_contours(fixed_mask, thickness=5)
        moving_contours, moving_edges = extract_contours(moving_mask, thickness=5)
        
        # Utiliser les contours pour la registration
        fixed_gray = cv2.cvtColor(fixed_contours, cv2.COLOR_RGB2GRAY).astype(np.float32)
        moving_gray = cv2.cvtColor(moving_contours, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # Inverser les valeurs pour que les contours soient "bright" (255) sur fond noir (0)
        # Car l'algorithme cherche à aligner les zones brillantes
        fixed_gray = 255 - fixed_gray
        moving_gray = 255 - moving_gray
        
    else:
        print("  Registration SimpleITK en cours (sur images segmentées)...")
        
        # Convertir en niveaux de gris pour la registration
        fixed_gray = cv2.cvtColor(fixed_img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
        moving_gray = cv2.cvtColor(moving_img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
    # Convertir en images SimpleITK
    fixed_sitk = sitk.GetImageFromArray(fixed_gray)
    moving_sitk = sitk.GetImageFromArray(moving_gray)
    
    # Initialiser la transformation affine
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_sitk,
        moving_sitk,
        sitk.AffineTransform(2),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    
    # Configurer la méthode de registration
    registration_method = sitk.ImageRegistrationMethod()
    
    # Métrique de similarité (Mutual Information pour images multimodales)
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)
    
    # Interpolateur
    registration_method.SetInterpolator(sitk.sitkLinear)
    
    # Optimiseur
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    # Multi-résolution
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Transformation initiale
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    
    # Exécuter la registration
    final_transform = registration_method.Execute(fixed_sitk, moving_sitk)
    
    print(f"    Optimiseur: {registration_method.GetOptimizerStopConditionDescription()}")
    
    # Appliquer la transformation à l'image originale (pas aux contours)
    # On veut recaler l'image complète, mais en utilisant la transformation calculée sur les contours
    registered_channels = []
    for channel_idx in range(3):
        channel_img = moving_img_np[:, :, channel_idx].astype(np.float32)
        channel_sitk = sitk.GetImageFromArray(channel_img)
        
        resampled = sitk.Resample(
            channel_sitk,
            fixed_sitk,
            final_transform,
            sitk.sitkLinear,
            0.0,
            channel_sitk.GetPixelID()
        )
        
        registered_channel = sitk.GetArrayFromImage(resampled)
        registered_channels.append(registered_channel)
    
    # Recombiner les canaux
    registered_img_np = np.stack(registered_channels, axis=-1).astype(np.uint8)
    
    print("  ✓ Registration terminée")
    
    # Si on a utilisé les contours, les retourner aussi
    if use_contours:
        # Appliquer la même transformation aux contours pour visualisation
        moving_contours_registered_channels = []
        for channel_idx in range(3):
            channel_img = moving_contours[:, :, channel_idx].astype(np.float32)
            channel_sitk = sitk.GetImageFromArray(channel_img)
            
            resampled = sitk.Resample(
                channel_sitk,
                fixed_sitk,
                final_transform,
                sitk.sitkLinear,
                255.0,  # Background blanc pour les contours
                channel_sitk.GetPixelID()
            )
            
            registered_channel = sitk.GetArrayFromImage(resampled)
            moving_contours_registered_channels.append(registered_channel)
        
        moving_contours_registered = np.stack(moving_contours_registered_channels, axis=-1).astype(np.uint8)
        
        return registered_img_np, final_transform, fixed_contours, moving_contours_registered
    else:
        return registered_img_np, final_transform, None, None

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
    
    # ÉTAPE 1: Segmentation Otsu sur les deux images
    print("  Application de la segmentation Otsu...")
    lowres_hes_segmented, mask_hes = apply_otsu_segmentation(lowres_hes_np)
    lowres_cd30_segmented, mask_cd30 = apply_otsu_segmentation(lowres_cd30_np)
    print("  ✓ Segmentation terminée")
    
    # ÉTAPE 2: Extraction des contours
    print("  Extraction des contours...")
    hes_contours, _ = extract_contours(mask_hes, thickness=5)
    cd30_contours_original, _ = extract_contours(mask_cd30, thickness=5)
    print("  ✓ Contours extraits")
    
    # ÉTAPE 3: Registration basée sur les CONTOURS
    print("  Démarrage de la registration Elastix sur les CONTOURS...")
    lowres_cd30_registered, transform_params, fixed_contours, moving_contours_registered = register_images_elastix(
        lowres_hes_segmented, 
        lowres_cd30_segmented,
        mask_hes,
        mask_cd30,
        use_contours=True
    )
    print("  ✓ CD30 recalé sur H&E en utilisant l'alignement des contours")
    
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
    
    # Créer des visualisations pour mettre en évidence la registration
    # 1. Overlays de contours (Rouge/Vert) - PRINCIPAL pour visualiser l'alignement des contours
    contours_overlay_no_reg = create_contour_overlay(hes_contours, cd30_contours_original)
    contours_overlay_registered = create_contour_overlay(fixed_contours, moving_contours_registered)
    
    # 2. Overlays colorés sur images segmentées (Magenta/Vert)
    overlay_no_reg = create_overlay(lowres_hes_segmented, lowres_cd30_segmented)
    overlay_registered = create_overlay(lowres_hes_segmented, lowres_cd30_registered)
    
    # 3. Damiers (Checkerboard)
    checkerboard_no_reg = create_checkerboard(lowres_hes_segmented, lowres_cd30_segmented, square_size=300)
    checkerboard_registered = create_checkerboard(lowres_hes_segmented, lowres_cd30_registered, square_size=300)
    
    # 4. Cartes de différence (heatmaps)
    diff_map_no_reg = create_difference_map(lowres_hes_segmented, lowres_cd30_segmented)
    diff_map_registered = create_difference_map(lowres_hes_segmented, lowres_cd30_registered)
    
    # Nettoyer
    slide_hes.close()
    slide_cd30.close()
    
    return {
        'hes': lowres_hes_segmented,
        'cd30_no_reg': lowres_cd30_segmented,
        'cd30_registered': lowres_cd30_registered,
        'hes_contours': hes_contours,
        'cd30_contours_original': cd30_contours_original,
        'cd30_contours_registered': moving_contours_registered,
        'contours_overlay_no_reg': contours_overlay_no_reg,
        'contours_overlay_registered': contours_overlay_registered,
        'overlay_no_reg': overlay_no_reg,
        'overlay_registered': overlay_registered,
        'checkerboard_no_reg': checkerboard_no_reg,
        'checkerboard_registered': checkerboard_registered,
        'diff_map_no_reg': diff_map_no_reg,
        'diff_map_registered': diff_map_registered,
        'n_regions': len(valid_regions)
    }


# Traiter tous les patients
print("="*80)
print("GÉNÉRATION DE LA FIGURE GLOBALE D'ALIGNEMENT")
print("="*80)

all_images = []
for patient_id in patient_ids:
    try:
        data = process_patient(patient_id)
        data['patient_id'] = patient_id
        all_images.append(data)
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

# Créer la grande figure - 3 RANGÉES pour mettre en évidence la registration
print("\n" + "="*80)
print("CRÉATION DE LA FIGURE FINALE")
print("="*80)

n_patients = len(all_images)
fig, axes = plt.subplots(3, 3, figsize=(20, 18))

for idx, data in enumerate(all_images):
    patient_id = data['patient_id']
    
    # RANGÉE 1: CONTOURS - Visualisation clé de l'alignement des bords
    axes[0, 0].imshow(data['hes_contours'])
    axes[0, 0].set_title(f"{patient_id} - H&E Contours (Reference)", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(data['contours_overlay_no_reg'])
    axes[0, 1].set_title("Contours BEFORE Registration\n(Red=H&E, Green=CD30, Yellow=Overlap)", 
                        fontsize=12, fontweight='bold', color='red')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(data['contours_overlay_registered'])
    axes[0, 2].set_title("Contours AFTER Registration\n(Red=H&E, Green=CD30, Yellow=Overlap)", 
                        fontsize=12, fontweight='bold', color='green')
    axes[0, 2].axis('off')
    
    # RANGÉE 2: Images segmentées complètes
    axes[1, 0].imshow(data['hes'])
    axes[1, 0].set_title("H&E Segmented", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(data['cd30_no_reg'])
    axes[1, 1].set_title("CD30 Segmented (Before)", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(data['cd30_registered'])
    axes[1, 2].set_title("CD30 Segmented (After)", fontsize=12, fontweight='bold', color='green')
    axes[1, 2].axis('off')
    
    # RANGÉE 3: Visualisations avancées
    axes[2, 0].imshow(data['checkerboard_no_reg'])
    axes[2, 0].set_title("Checkerboard: Before", fontsize=12, fontweight='bold', color='red')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(data['checkerboard_registered'])
    axes[2, 1].set_title("Checkerboard: After", fontsize=12, fontweight='bold', color='green')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(data['diff_map_registered'])
    axes[2, 2].set_title("Difference Map\n(Blue=Good, Red=Poor)", fontsize=12, fontweight='bold')
    axes[2, 2].axis('off')

plt.suptitle('Contour-Based Multi-Modal Image Registration (Otsu Segmentation)\nH&E to CD30 Alignment via Affine Transform on Edge Features', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

# Sauvegarder en haute qualité
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='jpg')
print(f"\n✓ Figure sauvegardée: {output_file}")
print(f"  Format: JPEG haute qualité (300 DPI)")
print(f"  Patients inclus: {', '.join([d['patient_id'] for d in all_images])}")
print("\n" + "="*80)
print("STRUCTURE DE LA FIGURE (3×3 grid) - REGISTRATION BASÉE SUR LES CONTOURS")
print("="*80)
print("  RANGÉE 1: CONTOURS - Visualisation de l'alignement des bords")
print("    - Contours H&E | Overlay AVANT | Overlay APRÈS")
print("    - Rouge=H&E, Vert=CD30, Jaune=Superposition parfaite")
print("")
print("  RANGÉE 2: Images segmentées complètes")
print("    - H&E segmenté | CD30 avant | CD30 après registration")
print("")
print("  RANGÉE 3: Visualisations de contrôle qualité")
print("    - Damier avant | Damier après | Carte de différence")
print("="*80)
print("\n✓ Registration basée sur les CONTOURS terminée!")
print("  Méthode: Alignment des bords tissulaires plutôt que des intensités internes")
print("  Avantage: Meilleure correspondance géométrique des structures")