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

def create_overlay(hes_img, cd30_img, alpha=0.5):
    """
    Crée une image overlay pour visualiser l'alignement.
    H&E en rouge, CD30 en cyan.
    """
    # S'assurer que les deux images ont la même taille
    if hes_img.shape != cd30_img.shape:
        # Redimensionner cd30 pour correspondre à hes
        cd30_img = cv2.resize(cd30_img, (hes_img.shape[1], hes_img.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    overlay = np.zeros_like(hes_img)
    overlay[:, :, 0] = (hes_img[:, :, 0] * alpha).astype(np.uint8)  # Rouge pour H&E
    overlay[:, :, 1] = (cd30_img[:, :, 1] * alpha).astype(np.uint8)  # Cyan pour CD30
    overlay[:, :, 2] = (cd30_img[:, :, 2] * alpha).astype(np.uint8)
    return overlay

def register_images_elastix(fixed_img_np, moving_img_np):
    """
    Recale l'image moving sur l'image fixed avec SimpleITK.
    Utilise ImageRegistrationMethod avec transformation affine.
    La registration se fait sur les images segmentées par Otsu.
    
    Args:
        fixed_img_np: Image fixe (H&E) en numpy array RGB
        moving_img_np: Image à recaler (CD30) en numpy array RGB
    
    Returns:
        registered_img_np: Image CD30 recalée en numpy array RGB
        transform: Transformation appliquée
    """
    print("  Registration SimpleITK en cours...")
    
    # Appliquer la segmentation Otsu sur les deux images
    fixed_mask = compute_tissue_mask(fixed_img_np)
    moving_mask = compute_tissue_mask(moving_img_np)
    
    # Appliquer les masques pour ne garder que les tissus
    fixed_segmented = fixed_img_np.copy()
    moving_segmented = moving_img_np.copy()
    fixed_segmented[fixed_mask == 0] = 255
    moving_segmented[moving_mask == 0] = 255
    
    # Convertir en niveaux de gris pour la registration (sur les images segmentées)
    fixed_gray = cv2.cvtColor(fixed_segmented, cv2.COLOR_RGB2GRAY).astype(np.float32)
    moving_gray = cv2.cvtColor(moving_segmented, cv2.COLOR_RGB2GRAY).astype(np.float32)
    
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
    
    # Segmenter les images avec Otsu pour l'affichage final
    mask_hes_display = compute_tissue_mask(lowres_hes_np)
    mask_cd30_registered = compute_tissue_mask(lowres_cd30_registered)
    
    # Créer les images segmentées
    img_hes_segmented = lowres_hes_np.copy()
    img_hes_segmented[mask_hes_display == 0] = 255
    
    img_cd30_segmented = lowres_cd30_registered.copy()
    img_cd30_segmented[mask_cd30_registered == 0] = 255
    
    # Créer les images annotées (seulement 2: HES et CD30 registered)
    img_hes_annotated = img_hes_segmented.copy()
    img_cd30_registered_annotated = img_cd30_segmented.copy()
    
    # Annoter les régions
    for region in valid_regions:
        # HES avec régions
        cv2.rectangle(
            img_hes_annotated,
            (region['x_lr'], region['y_lr']),
            (region['x_lr'] + region['size_lr'], region['y_lr'] + region['size_lr']),
            (255, 0, 0), 8
        )
        
        # CD30 registered avec régions
        region_x_cd30_lr = int(region['x'] / downsample_cd30)
        region_y_cd30_lr = int(region['y'] / downsample_cd30)
        region_size_cd30_lr = int(region_size / downsample_cd30)
        
        cv2.rectangle(
            img_cd30_registered_annotated,
            (region_x_cd30_lr, region_y_cd30_lr),
            (region_x_cd30_lr + region_size_cd30_lr, region_y_cd30_lr + region_size_cd30_lr),
            (0, 255, 0), 8
        )
    
    # Nettoyer
    slide_hes.close()
    slide_cd30.close()
    
    return (img_hes_annotated, img_cd30_registered_annotated, len(valid_regions))


# Traiter tous les patients
print("="*80)
print("GÉNÉRATION DE LA FIGURE GLOBALE D'ALIGNEMENT")
print("="*80)

all_images = []
for patient_id in patient_ids:
    try:
        img_hes, img_cd30_reg, n_regions = process_patient(patient_id)
        all_images.append({
            'patient_id': patient_id,
            'hes': img_hes,
            'cd30_registered': img_cd30_reg,
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

# Créer la grande figure - 2 colonnes seulement
print("\n" + "="*80)
print("CRÉATION DE LA FIGURE FINALE")
print("="*80)

n_patients = len(all_images)
fig, axes = plt.subplots(n_patients, 2, figsize=(16, 8*n_patients))

# S'assurer que axes est un tableau 2D même avec un seul patient
if n_patients == 1:
    axes = axes.reshape(1, -1)

for idx, data in enumerate(all_images):
    # Colonne 1: H&E segmenté + régions
    axes[idx, 0].imshow(data['hes'])
    axes[idx, 0].set_title(f"{data['patient_id']} - H&E segmenté (Otsu)\n{data['n_regions']} régions (boîtes rouges)", 
                           fontsize=14, fontweight='bold')
    axes[idx, 0].axis('off')
    
    # Colonne 2: CD30 segmenté + registration + régions
    axes[idx, 1].imshow(data['cd30_registered'])
    axes[idx, 1].set_title(f"CD30 segmenté (Otsu) + Registration\n{data['n_regions']} régions (boîtes vertes)", 
                           fontsize=14, fontweight='bold', color='green')
    axes[idx, 1].axis('off')

plt.suptitle('Images segmentées par Otsu - Registration CD30 → H&E', 
             fontsize=18, fontweight='bold', y=0.998)
plt.tight_layout()

# Sauvegarder en haute qualité
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='jpg')
print(f"\n✓ Figure sauvegardée: {output_file}")
print(f"  Format: JPEG haute qualité (300 DPI)")
print(f"  Patients inclus: {', '.join([d['patient_id'] for d in all_images])}")
print(f"  Total régions: {sum([d['n_regions'] for d in all_images])}")
print("\n" + "="*80)
print("STRUCTURE DE LA FIGURE (2 colonnes)")
print("="*80)
print("  Colonne 1: H&E segmenté (Otsu) avec régions (boîtes rouges)")
print("  Colonne 2: CD30 segmenté (Otsu) + Registration avec régions (boîtes vertes)")
print("  Note: La registration se fait sur les images segmentées par Otsu")
print("="*80)
print("\n✓ Visualisation terminée!")