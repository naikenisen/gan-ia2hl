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
region_size = 15000
stride_region = 15000
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

def compute_region_alignment(region_hes, region_cd30):
    """
    Calcule la transformation optimale (rotation + translation) entre deux régions.
    Retourne l'angle de rotation en degrés et le score de correspondance.
    """
    # Convertir en niveaux de gris
    gray_hes = cv2.cvtColor(region_hes, cv2.COLOR_RGB2GRAY)
    gray_cd30 = cv2.cvtColor(region_cd30, cv2.COLOR_RGB2GRAY)
    
    # Détecter les features avec ORB
    orb = cv2.ORB_create(nfeatures=1000)
    
    kp1, des1 = orb.detectAndCompute(gray_hes, None)
    kp2, des2 = orb.detectAndCompute(gray_cd30, None)
    
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return 0.0, 0.0  # Pas assez de features
    
    # Matcher les features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    if len(matches) < 10:
        return 0.0, 0.0  # Pas assez de correspondances
    
    # Trier les matches par distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extraire les points correspondants
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:100]]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:100]]).reshape(-1, 2)
    
    try:
        # Estimer la transformation affine avec RANSAC
        model, inliers = ransac(
            (pts1, pts2),
            AffineTransform,
            min_samples=3,
            residual_threshold=10,
            max_trials=1000
        )
        
        # Extraire l'angle de rotation de la matrice affine
        rotation_matrix = model.params[:2, :2]
        angle_rad = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        angle_deg = np.degrees(angle_rad)
        
        # Score de correspondance = ratio d'inliers
        score = np.sum(inliers) / len(matches)
        
        return angle_deg, score
        
    except:
        return 0.0, 0.0

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
            
            # Extraire les régions pour calculer l'alignement
            region_hes_img = lowres_hes_np[region_y_lr:region_y_lr+region_size_lr, 
                                           region_x_lr:region_x_lr+region_size_lr]
            region_cd30_img = lowres_cd30_np[region_y_cd30_lr:region_y_cd30_lr+region_size_cd30_lr,
                                             region_x_cd30_lr:region_x_cd30_lr+region_size_cd30_lr]
            
            # Calculer la rotation optimale
            angle, score = compute_region_alignment(region_hes_img, region_cd30_img)
            
            valid_regions.append({
                'index': region_index,
                'x': region_x,
                'y': region_y,
                'x_lr': region_x_lr,
                'y_lr': region_y_lr,
                'size_lr': region_size_lr,
                'rotation_angle': angle,
                'alignment_score': score
            })
            region_index += 1
    
    print(f"  Trouvé {len(valid_regions)} régions valides")
    if len(valid_regions) > 0:
        avg_score = np.mean([r['alignment_score'] for r in valid_regions])
        print(f"  Score moyen d'alignement: {avg_score:.3f}")
    
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
    
    # Générer des couleurs distinctes pour chaque région
    np.random.seed(42)  # Pour reproductibilité
    colors = []
    for i in range(len(valid_regions)):
        # Générer des couleurs vives et distinctes
        hue = int(i * 360 / len(valid_regions))
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)[0][0]
        colors.append(tuple(int(c) for c in color_rgb))
    
    # Stocker les couleurs dans les régions pour la légende
    for idx, region in enumerate(valid_regions):
        region['color'] = colors[idx]
    
    # Annoter les régions avec information de rotation
    for idx, region in enumerate(valid_regions):
        color = colors[idx]
        
        # HES avec rectangles colorés
        cv2.rectangle(
            img_hes_annotated,
            (region['x_lr'], region['y_lr']),
            (region['x_lr'] + region['size_lr'], region['y_lr'] + region['size_lr']),
            color, 8
        )
        
        # Ajouter le numéro de région
        region_text = f"R{idx+1}"
        cv2.putText(
            img_hes_annotated,
            region_text,
            (region['x_lr'] + 10, region['y_lr'] + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            4
        )
        
        # Ajouter l'angle de rotation
        angle_text = f"{region['rotation_angle']:.1f}°"
        cv2.putText(
            img_hes_annotated,
            angle_text,
            (region['x_lr'] + 10, region['y_lr'] + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            3
        )
        
        # CD30 avec rectangles pivotés de la même couleur
        region_x_cd30_lr = int(region['x'] / downsample_cd30)
        region_y_cd30_lr = int(region['y'] / downsample_cd30)
        region_size_cd30_lr = int(region_size / downsample_cd30)
        
        # Dessiner un rectangle pivoté pour visualiser la rotation
        center = (region_x_cd30_lr + region_size_cd30_lr // 2, 
                 region_y_cd30_lr + region_size_cd30_lr // 2)
        box = cv2.boxPoints(((center[0], center[1]), 
                             (region_size_cd30_lr, region_size_cd30_lr), 
                             region['rotation_angle']))
        box = np.int0(box)
        cv2.drawContours(img_cd30_annotated, [box], 0, color, 8)
        
        # Ajouter le numéro de région
        cv2.putText(
            img_cd30_annotated,
            region_text,
            (region_x_cd30_lr + 10, region_y_cd30_lr + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            4
        )
        
        # Ajouter le score d'alignement
        score_text = f"S:{region['alignment_score']:.2f}"
        cv2.putText(
            img_cd30_annotated,
            score_text,
            (region_x_cd30_lr + 10, region_y_cd30_lr + 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            3
        )
    
    # Nettoyer
    slide_hes.close()
    slide_cd30.close()
    
    return img_hes_original, img_hes_segmented, img_cd30_segmented, img_hes_annotated, img_cd30_annotated, len(valid_regions), valid_regions


# Traiter tous les patients
print("="*80)
print("GÉNÉRATION DE LA FIGURE GLOBALE D'ALIGNEMENT")
print("="*80)

all_images = []
for patient_id in patient_ids:
    try:
        img_hes_orig, img_hes_seg, img_cd30_seg, img_hes_rect, img_cd30_rect, n_regions, regions = process_patient(patient_id)
        all_images.append({
            'patient_id': patient_id,
            'hes_original': img_hes_orig,
            'hes_segmented': img_hes_seg,
            'cd30_segmented': img_cd30_seg,
            'hes_rectangles': img_hes_rect,
            'cd30_rectangles': img_cd30_rect,
            'n_regions': n_regions,
            'regions': regions
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
    
    # Colonne 4: HES avec rectangles colorés
    axes[idx, 3].imshow(data['hes_rectangles'])
    axes[idx, 3].set_title(f"H&E + Régions\n({data['n_regions']} régions)", 
                           fontsize=12, fontweight='bold')
    axes[idx, 3].axis('off')
    
    # Colonne 5: CD30 avec rectangles colorés (mêmes couleurs)
    axes[idx, 4].imshow(data['cd30_rectangles'])
    axes[idx, 4].set_title(f"CD30 + Régions\n({data['n_regions']} régions)", 
                           fontsize=12, fontweight='bold')
    axes[idx, 4].axis('off')
    
    # Ajouter une légende de couleurs sous les images
    legend_text = "Légende: "
    for i, region in enumerate(data['regions'][:5]):  # Limiter à 5 régions pour la légende
        color_norm = tuple(c/255.0 for c in region['color'])
        legend_text += f"R{i+1} "
        # Ajouter un patch de couleur
        rect = patches.Rectangle((0.02 + i*0.18, 0.02), 0.15, 0.05, 
                                 transform=axes[idx, 4].transAxes,
                                 facecolor=color_norm, edgecolor='black', linewidth=2)
        axes[idx, 4].add_patch(rect)
        axes[idx, 4].text(0.095 + i*0.18, 0.045, f'R{i+1}', 
                         transform=axes[idx, 4].transAxes,
                         ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    if len(data['regions']) > 5:
        axes[idx, 4].text(0.92, 0.045, f'+{len(data["regions"])-5}', 
                         transform=axes[idx, 4].transAxes,
                         ha='center', va='center', fontsize=10, fontweight='bold')

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
