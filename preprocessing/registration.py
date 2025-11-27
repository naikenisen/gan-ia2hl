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

def register_images_elastix(fixed_img_np, moving_img_np):
    """
    Recale l'image moving sur l'image fixed avec SimpleITK.
    Utilise ImageRegistrationMethod avec transformation affine.
    
    Args:
        fixed_img_np: Image fixe (H&E) en numpy array RGB
        moving_img_np: Image à recaler (CD30) en numpy array RGB
    
    Returns:
        registered_img_np: Image CD30 recalée en numpy array RGB
        transform: Transformation appliquée
    """
    print("  Registration SimpleITK en cours...")
    
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
    
    # Appliquer la transformation à chaque canal RGB
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
    
    return registered_img_np, final_transform

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
    
    # ÉTAPE DE REGISTRATION: Recaler CD30 sur H&E avec Elastix
    print("  Démarrage de la registration Elastix...")
    lowres_cd30_registered, transform_params = register_images_elastix(lowres_hes_np, lowres_cd30_np)
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
    
    # Créer des visualisations pour mettre en évidence la registration
    # 1. Overlays colorés (Magenta/Vert)
    overlay_no_reg = create_overlay(lowres_hes_np, lowres_cd30_np)
    overlay_registered = create_overlay(lowres_hes_np, lowres_cd30_registered)
    
    # 2. Damiers (Checkerboard)
    checkerboard_no_reg = create_checkerboard(lowres_hes_np, lowres_cd30_np, square_size=300)
    checkerboard_registered = create_checkerboard(lowres_hes_np, lowres_cd30_registered, square_size=300)
    
    # 3. Cartes de différence (heatmaps)
    diff_map_no_reg = create_difference_map(lowres_hes_np, lowres_cd30_np)
    diff_map_registered = create_difference_map(lowres_hes_np, lowres_cd30_registered)
    
    # Nettoyer
    slide_hes.close()
    slide_cd30.close()
    
    return {
        'hes': lowres_hes_np,
        'cd30_no_reg': lowres_cd30_np,
        'cd30_registered': lowres_cd30_registered,
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
    
    # RANGÉE 1: Images sources
    axes[0, 0].imshow(data['hes'])
    axes[0, 0].set_title(f"{patient_id} - H&E (Reference)", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(data['cd30_no_reg'])
    axes[0, 1].set_title("CD30 (Unregistered)", fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(data['cd30_registered'])
    axes[0, 2].set_title("CD30 (Registered)", fontsize=14, fontweight='bold', color='green')
    axes[0, 2].axis('off')
    
    # RANGÉE 2: Overlays colorés (mise en évidence de l'alignement)
    axes[1, 0].imshow(data['overlay_no_reg'])
    axes[1, 0].set_title("Before Registration\n(Magenta=H&E, Green=CD30)", fontsize=12, fontweight='bold', color='red')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(data['overlay_registered'])
    axes[1, 1].set_title("After Registration\n(Magenta=H&E, Green=CD30)", fontsize=12, fontweight='bold', color='green')
    axes[1, 1].axis('off')
    
    # Texte explicatif au centre
    axes[1, 2].text(0.5, 0.5, "Better alignment\n→ More yellow/white\n\nMisalignment\n→ Magenta or Green", 
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
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

plt.suptitle('Multi-Modal Image Registration: H&E to CD30 Alignment\nAffine Transformation via Mattes Mutual Information', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

# Sauvegarder en haute qualité
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='jpg')
print(f"\n✓ Figure sauvegardée: {output_file}")
print(f"  Format: JPEG haute qualité (300 DPI)")
print(f"  Patients inclus: {', '.join([d['patient_id'] for d in all_images])}")
print("\n" + "="*80)
print("STRUCTURE DE LA FIGURE (3×3 grid)")
print("="*80)
print("  RANGÉE 1: Images sources")
print("    - H&E référence | CD30 brut | CD30 recalé")
print("")
print("  RANGÉE 2: Overlays colorés (IMPACT VISUEL MAXIMAL)")
print("    - Avant registration | Après registration | Légende")
print("    - Magenta + Vert = Jaune/Blanc (bon alignement)")
print("")
print("  RANGÉE 3: Visualisations techniques")
print("    - Damier avant | Damier après | Carte de différence")
print("="*80)
print("\n✓ Visualisation terminée!")