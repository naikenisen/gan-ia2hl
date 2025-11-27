import os
import openslide
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.feature import ORB
from skimage.measure import ransac
from skimage.transform import AffineTransform
import warnings
warnings.filterwarnings('ignore')

# Configuration
input_folder = "/gold/data_feasibility"
output_file = "alignment_overview.jpg"

# Liste des patients à traiter
patient_ids = ["AHL002","AHL006"]

patch_size = 2000
region_size = 12000
stride_region = 12000
lowres_level = 2
tissue_threshold = 0.20

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
    
    # Calculer les masques de tissu
    mask_hes = compute_tissue_mask(lowres_hes)
    mask_cd30 = compute_tissue_mask(lowres_cd30)
    
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
    
    # Créer les différentes versions d'images
    img_hes_original = lowres_hes_np.copy()
    img_cd30_original = lowres_cd30_np.copy()
    
    # Créer un overlay du masque de segmentation pour H&E
    mask_hes_colored = np.zeros_like(lowres_hes_np)
    mask_hes_colored[:, :, 1] = mask_hes  # Canal vert pour le masque
    img_hes_segmented = cv2.addWeighted(lowres_hes_np, 0.7, mask_hes_colored, 0.3, 0)
    
    # Créer un overlay du masque de segmentation pour CD30
    mask_cd30_colored = np.zeros_like(lowres_cd30_np)
    mask_cd30_colored[:, :, 1] = mask_cd30  # Canal vert pour le masque
    img_cd30_segmented = cv2.addWeighted(lowres_cd30_np, 0.7, mask_cd30_colored, 0.3, 0)
    
    # Images avec rectangles
    img_hes_annotated = lowres_hes_np.copy()
    img_cd30_annotated = lowres_cd30_np.copy()
    
    # Annoter les régions
    for region in valid_regions:
        # HES avec rectangles
        cv2.rectangle(
            img_hes_annotated,
            (region['x_lr'], region['y_lr']),
            (region['x_lr'] + region['size_lr'], region['y_lr'] + region['size_lr']),
            (255, 0, 0), 8
        )
        
        # CD30 avec rectangles
        region_x_cd30_lr = int(region['x'] / downsample_cd30)
        region_y_cd30_lr = int(region['y'] / downsample_cd30)
        region_size_cd30_lr = int(region_size / downsample_cd30)
        
        cv2.rectangle(
            img_cd30_annotated,
            (region_x_cd30_lr, region_y_cd30_lr),
            (region_x_cd30_lr + region_size_cd30_lr, region_y_cd30_lr + region_size_cd30_lr),
            (0, 0, 255), 8
        )
    
    # Nettoyer
    slide_hes.close()
    slide_cd30.close()
    
    return img_hes_original, img_hes_segmented, img_cd30_segmented, img_hes_annotated, img_cd30_annotated, len(valid_regions)


# Traiter tous les patients
print("="*80)
print("GÉNÉRATION DE LA FIGURE GLOBALE D'ALIGNEMENT")
print("="*80)

all_images = []
for patient_id in patient_ids:
    try:
        img_hes_orig, img_hes_seg, img_cd30_seg, img_hes_rect, img_cd30_rect, n_regions = process_patient(patient_id)
        all_images.append({
            'patient_id': patient_id,
            'hes_original': img_hes_orig,
            'hes_segmented': img_hes_seg,
            'cd30_segmented': img_cd30_seg,
            'hes_rectangles': img_hes_rect,
            'cd30_rectangles': img_cd30_rect,
            'n_regions': n_regions
        })
    except Exception as e:
        print(f"  ⚠ Erreur pour {patient_id}: {e}")
        continue

# Créer la grande figure
print("\n" + "="*80)
print("CRÉATION DE LA FIGURE FINALE")
print("="*80)

n_patients = len(all_images)
fig, axes = plt.subplots(n_patients, 5, figsize=(30, 5*n_patients))

# S'assurer que axes est un tableau 2D même avec un seul patient
if n_patients == 1:
    axes = axes.reshape(1, -1)

for idx, data in enumerate(all_images):
    # Colonne 1: HES original
    axes[idx, 0].imshow(data['hes_original'])
    axes[idx, 0].set_title(f"{data['patient_id']} - H&E Original", 
                           fontsize=12, fontweight='bold')
    axes[idx, 0].axis('off')
    
    # Colonne 2: HES avec segmentation
    axes[idx, 1].imshow(data['hes_segmented'])
    axes[idx, 1].set_title(f"H&E + Segmentation\n(masque vert 30%)", 
                           fontsize=12, fontweight='bold')
    axes[idx, 1].axis('off')
    
    # Colonne 3: CD30 avec segmentation
    axes[idx, 2].imshow(data['cd30_segmented'])
    axes[idx, 2].set_title(f"CD30 + Segmentation\n(masque vert 30%)", 
                           fontsize=12, fontweight='bold')
    axes[idx, 2].axis('off')
    
    # Colonne 4: HES avec rectangles rouges
    axes[idx, 3].imshow(data['hes_rectangles'])
    axes[idx, 3].set_title(f"H&E + Régions\n({data['n_regions']} rectangles rouges)", 
                           fontsize=12, fontweight='bold')
    axes[idx, 3].axis('off')
    
    # Colonne 5: CD30 avec rectangles bleus
    axes[idx, 4].imshow(data['cd30_rectangles'])
    axes[idx, 4].set_title(f"CD30 + Régions\n({data['n_regions']} rectangles bleus)", 
                           fontsize=12, fontweight='bold')
    axes[idx, 4].axis('off')

plt.suptitle('Pipeline complet: Segmentation et Alignement des lames H&E et CD30', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()

# Sauvegarder en haute qualité
plt.savefig(output_file, dpi=500, bbox_inches='tight', format='jpg')
print(f"\n✓ Figure sauvegardée: {output_file}")
print(f"  Format: JPEG haute qualité (500 DPI)")
print(f"  Patients inclus: {', '.join([d['patient_id'] for d in all_images])}")
print(f"  Total régions: {sum([d['n_regions'] for d in all_images])}")
print("\n✓ Visualisation terminée!")
