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
import wandb
import pandas as pd
from collections import defaultdict
warnings.filterwarnings('ignore')
wandb.login(key="ab67e0f4c27fad7a0d47405f84a8a4deb80056ba")

input_folder = "/gold/data_feasibility"
output_folder = "./visualizations_ORB"
os.makedirs(output_folder, exist_ok=True)

registration_metrics = defaultdict(lambda: defaultdict(dict))

base_region_size = 12000  # Taille de référence, sera ajustée par slide
overlap_percent = 0.10    # 10% de chevauchement entre les régions
lowres_level = 2

# Initialiser wandb
wandb.init(
    project="ia2hl-preprocessing",
    name="akaze-mask-based-registration-80pct",
    config={
        "feature_detector": "AKAZE",
        "registration_method": "mask_based",
        "mask_enhancement": True,
        "base_region_size": base_region_size,
        "adaptive_regions": True,
        "overlap_percent": overlap_percent,
        "lowres_level": lowres_level,
        "input_folder": input_folder,
        "output_folder": output_folder,
        "min_inliers": 20,
        "min_tissue_percent": 0.80,
        "spatial_coherence": True,
        "global_registration": "tissue_mask_enhanced"
    }
)

def compute_tissue_mask(img_rgb):
    img_np = np.array(img_rgb)

    # Conversion RGB -> HSV
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

    # Normalisation de l'histogramme sur la saturation
    sat = hsv[:, :, 1]
    sat_eq = cv2.equalizeHist(sat)

    # --- Ajout du flou gaussien AVANT le seuillage ---
    sat_blur = cv2.GaussianBlur(sat_eq, (15, 15), 2)

    # Seuillage (Otsu) directement sur l'image floutée
    _, mask = cv2.threshold(sat_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphologie
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def perform_region_registration(img_fixed, img_moving, patient_id, region_idx):
    """
    Effectue la registration d'une région sans générer de figure.
    Retourne les métriques de registration et la transformation.
    """
    # Conversion en niveaux de gris
    gray_fixed = cv2.cvtColor(img_fixed, cv2.COLOR_RGB2GRAY)
    gray_moving = cv2.cvtColor(img_moving, cv2.COLOR_RGB2GRAY)
    
    # Détection AKAZE (meilleur que ORB pour l'histologie)
    akaze = cv2.AKAZE_create()
    kp1, desc1 = akaze.detectAndCompute(gray_fixed, None)
    kp2, desc2 = akaze.detectAndCompute(gray_moving, None)
    
    # Logger la détection des keypoints
    wandb.log({
        f"{patient_id}/region_{region_idx}/keypoints_hes": len(kp1) if kp1 else 0,
        f"{patient_id}/region_{region_idx}/keypoints_cd30": len(kp2) if kp2 else 0,
    })
    
    if desc1 is None or desc2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("Pas assez de points détectés")
        wandb.log({
            f"{patient_id}/region_{region_idx}/status": "failed_detection"
        })
        return {
            'inliers': 0,
            'total_matches': 0,
            'ratio': 0.0,
            'status': 'failed_detection',
            'transform': None
        }
    
    # Matching (AKAZE utilise des descripteurs binaires comme ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # AKAZE produit généralement plus de matches de qualité, on peut être plus sélectif
    num_good_matches = min(len(matches), max(50, int(len(matches) * 0.20)))
    good_matches = matches[:num_good_matches]
    
    # Logger les correspondances
    wandb.log({
        f"{patient_id}/region_{region_idx}/total_matches": len(matches),
        f"{patient_id}/region_{region_idx}/good_matches": len(good_matches),
    })
    
    if len(good_matches) < 4:
        print("Pas assez de correspondances")
        wandb.log({
            f"{patient_id}/region_{region_idx}/status": "failed_matching"
        })
        return {
            'inliers': 0,
            'total_matches': len(matches),
            'ratio': 0.0,
            'status': 'failed_matching',
            'transform': None
        }
    
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
        
        # Seuil minimum de 20 inliers pour valider la registration
        MIN_INLIERS = 20
        
        if model is None or np.sum(inliers) < MIN_INLIERS:
            num_inliers_found = np.sum(inliers) if inliers is not None else 0
            print(f"RANSAC échoué ou insuffisant: {num_inliers_found} inliers (minimum requis: {MIN_INLIERS})")
            wandb.log({
                f"{patient_id}/region_{region_idx}/status": "failed_ransac_insufficient_inliers",
                f"{patient_id}/region_{region_idx}/inliers_found": int(num_inliers_found),
                f"{patient_id}/region_{region_idx}/min_required": MIN_INLIERS
            })
            return {
                'inliers': int(num_inliers_found),
                'total_matches': len(good_matches),
                'ratio': 0.0,
                'status': f'failed_insufficient_inliers',
                'transform': None
            }
        
        num_inliers = np.sum(inliers)
        inlier_ratio = num_inliers / len(good_matches)
        
        print(f"✓ {num_inliers} inliers sur {len(good_matches)} matches (seuil: {MIN_INLIERS})")
        
        # Logger les résultats RANSAC
        wandb.log({
            f"{patient_id}/region_{region_idx}/inliers": int(num_inliers),
            f"{patient_id}/region_{region_idx}/inlier_ratio": inlier_ratio,
            f"{patient_id}/region_{region_idx}/status": "success"
        })
        
        return {
            'inliers': int(num_inliers),
            'total_matches': len(good_matches),
            'ratio': float(inlier_ratio),
            'status': 'success',
            'transform': model
        }
        
    except Exception as e:
        print(f"Erreur: {e}")
        wandb.log({
            f"{patient_id}/region_{region_idx}/status": "failed_error",
            f"{patient_id}/region_{region_idx}/error": str(e)
        })
        return {
            'inliers': 0,
            'total_matches': 0,
            'ratio': 0.0,
            'status': 'failed_error',
            'error': str(e),
            'transform': None
        }



def interpolate_transform_from_neighbors(region_info, regions_info):
    """
    Interpole la transformation d'une région à partir de ses voisins valides.
    Utilise une moyenne pondérée par la distance inverse.
    """
    grid_x = region_info['x_lr'] // region_info['size_lr']
    grid_y = region_info['y_lr'] // region_info['size_lr']
    
    # Trouver tous les voisins avec transformation valide
    valid_neighbors = []
    for other in regions_info:
        if other['metrics']['transform'] is None:
            continue
        
        other_grid_x = other['x_lr'] // other['size_lr']
        other_grid_y = other['y_lr'] // other['size_lr']
        
        # Distance dans la grille
        dist = np.sqrt((grid_x - other_grid_x)**2 + (grid_y - other_grid_y)**2)
        
        # Prendre les voisins dans un rayon de 3 cellules
        if 0 < dist <= 3:
            valid_neighbors.append({
                'transform': other['metrics']['transform'],
                'distance': dist,
                'weight': 1.0 / dist  # Poids = inverse de la distance
            })
    
    if not valid_neighbors:
        return None
    
    # Normaliser les poids
    total_weight = sum(n['weight'] for n in valid_neighbors)
    for n in valid_neighbors:
        n['weight'] /= total_weight
    
    # Interpoler la matrice de transformation (moyenne pondérée)
    interpolated_params = np.zeros((3, 3))
    for neighbor in valid_neighbors:
        interpolated_params += neighbor['weight'] * neighbor['transform'].params
    
    # Créer une nouvelle transformation
    interpolated_transform = AffineTransform(matrix=interpolated_params)
    
    return interpolated_transform


def apply_spatial_coherence_correction(regions_info, patient_id):
    """
    Applique la correction par cohérence spatiale :
    - Identifie les régions qui ont échoué
    - Interpole leur transformation à partir des voisins valides
    """
    failed_regions = [r for r in regions_info if r['metrics']['status'] != 'success']
    success_regions = [r for r in regions_info if r['metrics']['status'] == 'success']
    
    if not success_regions:
        print("  ⚠ Aucune région valide pour interpolation")
        return regions_info
    
    corrected_count = 0
    
    for region in failed_regions:
        print(f"  → Tentative d'interpolation pour région {region['idx']}...")
        
        interpolated_transform = interpolate_transform_from_neighbors(region, success_regions)
        
        if interpolated_transform is not None:
            # Mettre à jour les métriques
            region['metrics']['transform'] = interpolated_transform
            region['metrics']['status'] = 'interpolated'
            region['metrics']['inliers'] = -1  # Marqueur spécial
            region['metrics']['ratio'] = -1.0
            
            corrected_count += 1
            print(f"    ✓ Transformation interpolée à partir de {len([r for r in success_regions if r['metrics']['transform'] is not None])} voisins")
            
            # Logger dans wandb
            wandb.log({
                f"{patient_id}/region_{region['idx']}/status": "interpolated",
                f"{patient_id}/region_{region['idx']}/correction": "spatial_coherence"
            })
        else:
            print(f"    ✗ Pas assez de voisins valides")
    
    if corrected_count > 0:
        print(f"\n  ✓ {corrected_count} région(s) corrigée(s) par interpolation spatiale")
        wandb.log({
            f"{patient_id}/spatial_coherence/corrected_regions": corrected_count,
            f"{patient_id}/spatial_coherence/failed_regions": len(failed_regions)
        })
    
    return regions_info


def generate_regions_triptych_figure(patient_id, regions_data):
    """
    Génère une grande figure avec toutes les sous-régions en triptyque :
    HES | CD30 originale | CD30 alignée
    """
    num_regions = len(regions_data)
    if num_regions == 0:
        return None
    
    # Calculer la disposition de la grille (3 colonnes par région)
    ncols = 3  # HES, CD30 originale, CD30 alignée
    nrows = num_regions
    
    # Créer une grande figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * num_regions))
    
    # Si une seule région, axes n'est pas un array 2D
    if num_regions == 1:
        axes = axes.reshape(1, -1)
    
    for idx, region_data in enumerate(regions_data):
        region_idx = region_data['idx']
        img_hes = region_data['img_hes']
        img_cd30_original = region_data['img_cd30_original']
        img_cd30_aligned = region_data['img_cd30_aligned']
        metrics = region_data['metrics']
        
        # Colonne 1 : HES
        axes[idx, 0].imshow(img_hes)
        axes[idx, 0].set_title(f'Région {region_idx} - HES (référence)', 
                               fontsize=12, fontweight='bold')
        axes[idx, 0].axis('off')
        
        # Colonne 2 : CD30 originale
        axes[idx, 1].imshow(img_cd30_original)
        axes[idx, 1].set_title(f'CD30 (originale)', 
                               fontsize=12, fontweight='bold')
        axes[idx, 1].axis('off')
        
        # Colonne 3 : CD30 alignée
        axes[idx, 2].imshow(img_cd30_aligned)
        
        # Titre selon le statut
        if metrics['status'] == 'success':
            title = f'CD30 (alignée)\n{metrics["inliers"]} inliers, ratio: {metrics["ratio"]:.3f}'
            color = 'green'
        elif metrics['status'] == 'interpolated':
            title = f'CD30 (alignée par interpolation)'
            color = 'orange'
        else:
            title = f'CD30 (échec)\nTransformation globale uniquement'
            color = 'red'
        
        axes[idx, 2].set_title(title, fontsize=12, fontweight='bold', color=color)
        axes[idx, 2].axis('off')
    
    plt.suptitle(f'Patient {patient_id} - Toutes les régions en triptyque', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Sauvegarder
    output_path = os.path.join(output_folder, f'{patient_id}_regions_triptych.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Triptyque des régions sauvegardé: {output_path}")
    
    # Logger dans wandb
    wandb.log({
        f"{patient_id}/triptych": wandb.Image(output_path)
    })
    
    return output_path


def generate_patient_overview_figure(lowres_hes_np, patient_id, regions_info):
    """
    Génère une figure unique par patient montrant toutes les régions sur la lame HES
    avec les métriques de registration affichées dans chaque région.
    """
    fig, ax = plt.subplots(1, 1, figsize=(20, 16))
    
    # Afficher l'image HES basse résolution
    ax.imshow(lowres_hes_np)
    ax.set_title(f'Patient {patient_id} - Vue d\'ensemble des régions avec métriques de registration', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Parcourir toutes les régions et dessiner les rectangles avec métriques
    for region_info in regions_info:
        region_idx = region_info['idx']
        x_lr = region_info['x_lr']
        y_lr = region_info['y_lr']
        size_lr = region_info['size_lr']
        metrics = region_info['metrics']
        
        # Couleur du rectangle selon le statut
        if metrics['status'] == 'success':
            edge_color = 'lime'
            face_color = 'lime'
            alpha_face = 0.1
            linewidth = 3
        elif metrics['status'] == 'interpolated':
            edge_color = 'orange'
            face_color = 'orange'
            alpha_face = 0.15
            linewidth = 2.5
        else:
            edge_color = 'red'
            face_color = 'red'
            alpha_face = 0.15
            linewidth = 2
        
        # Dessiner le rectangle de la région
        rect = patches.Rectangle(
            (x_lr, y_lr), 
            size_lr, 
            size_lr,
            linewidth=linewidth,
            edgecolor=edge_color,
            facecolor=face_color,
            alpha=alpha_face
        )
        ax.add_patch(rect)
        
        # Préparer le texte à afficher
        if metrics['status'] == 'success':
            text = f"Région {region_idx}\n{metrics['inliers']} inliers\nRatio: {metrics['ratio']:.3f}"
            text_color = 'white'
            bbox_color = 'green'
        elif metrics['status'] == 'interpolated':
            text = f"Région {region_idx}\nInterpolée"
            text_color = 'white'
            bbox_color = 'darkorange'
        else:
            text = f"Région {region_idx}\nÉchec"
            text_color = 'white'
            bbox_color = 'darkred'
        
        # Position du texte au centre de la région
        text_x = x_lr + size_lr / 2
        text_y = y_lr + size_lr / 2
        
        # Afficher le texte avec un fond
        ax.text(
            text_x, 
            text_y, 
            text,
            fontsize=12,
            fontweight='bold',
            color=text_color,
            ha='center',
            va='center',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor=bbox_color,
                alpha=0.8,
                edgecolor='white',
                linewidth=2
            )
        )
    
    plt.tight_layout()
    
    # Sauvegarder la figure
    output_path = os.path.join(output_folder, f'{patient_id}_regions_overview.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Vue d'ensemble sauvegardée: {output_path}")
    
    # Logger dans wandb
    wandb.log({
        f"{patient_id}/overview": wandb.Image(output_path)
    })
    
    return output_path


def register_whole_slide(lowres_hes_np, lowres_cd30_np, patient_id):
    """
    Effectue une registration globale de la lame entière à basse résolution.
    Utilise les masques de tissu pour une registration plus robuste.
    Retourne la transformation globale et l'image CD30 alignée.
    """
    print("\n[REGISTRATION GLOBALE] Alignement de la lame entière...")
    
    # Calculer les masques de tissu pour les deux lames
    print("  → Calcul des masques de tissu...")
    mask_hes = compute_tissue_mask(lowres_hes_np)
    mask_cd30 = compute_tissue_mask(lowres_cd30_np)
    
    # Améliorer les masques pour la détection de points d'intérêt
    enhanced_mask_hes = enhance_mask_for_registration(mask_hes)
    enhanced_mask_cd30 = enhance_mask_for_registration(mask_cd30)
    
    # Conversion en niveaux de gris des images originales
    gray_hes = cv2.cvtColor(lowres_hes_np, cv2.COLOR_RGB2GRAY)
    gray_cd30 = cv2.cvtColor(lowres_cd30_np, cv2.COLOR_RGB2GRAY)
    
    # Détection AKAZE - Trois approches combinées
    akaze = cv2.AKAZE_create()
    
    # Option 1: Détection sur masques améliorés (contours + structure)
    print("  → Détection AKAZE sur les masques améliorés...")
    kp1_mask, desc1_mask = akaze.detectAndCompute(enhanced_mask_hes, None)
    kp2_mask, desc2_mask = akaze.detectAndCompute(enhanced_mask_cd30, None)
    
    # Option 2: Détection sur images grises mais SEULEMENT dans les zones de tissu
    print("  → Détection AKAZE sur images avec masques de tissu...")
    kp1_masked, desc1_masked = akaze.detectAndCompute(gray_hes, mask=mask_hes)
    kp2_masked, desc2_masked = akaze.detectAndCompute(gray_cd30, mask=mask_cd30)
    
    # Option 3: Détection sur masques binaires simples (pour la forme générale)
    print("  → Détection AKAZE sur masques binaires...")
    kp1_binary, desc1_binary = akaze.detectAndCompute(mask_hes, None)
    kp2_binary, desc2_binary = akaze.detectAndCompute(mask_cd30, None)
    
    # Combiner les trois approches pour plus de robustesse
    kp1_all = []
    kp2_all = []
    desc1_all = []
    desc2_all = []
    
    # Ajouter les keypoints et descripteurs de chaque méthode
    methods = [
        ("enhanced", kp1_mask, desc1_mask, kp2_mask, desc2_mask),
        ("masked", kp1_masked, desc1_masked, kp2_masked, desc2_masked),
        ("binary", kp1_binary, desc1_binary, kp2_binary, desc2_binary)
    ]
    
    for method_name, kp1, desc1, kp2, desc2 in methods:
        if kp1 and desc1 is not None and kp2 and desc2 is not None:
            kp1_all.extend(kp1)
            kp2_all.extend(kp2)
            desc1_all.append(desc1)
            desc2_all.append(desc2)
            print(f"    {method_name}: {len(kp1)} points HES, {len(kp2)} points CD30")
    
    # Combiner tous les descripteurs
    if desc1_all and desc2_all:
        desc1_combined = np.vstack(desc1_all)
        desc2_combined = np.vstack(desc2_all)
        kp1_combined = kp1_all
        kp2_combined = kp2_all
    else:
        desc1_combined = None
        desc2_combined = None
        kp1_combined = []
        kp2_combined = []
    
    print(f"  Points combinés - HES: {len(kp1_combined)}, CD30: {len(kp2_combined)}")
    
    # Logger les keypoints par méthode
    wandb.log({
        f"{patient_id}/global/keypoints_hes_enhanced": len(kp1_mask) if kp1_mask else 0,
        f"{patient_id}/global/keypoints_cd30_enhanced": len(kp2_mask) if kp2_mask else 0,
        f"{patient_id}/global/keypoints_hes_masked": len(kp1_masked) if kp1_masked else 0,
        f"{patient_id}/global/keypoints_cd30_masked": len(kp2_masked) if kp2_masked else 0,
        f"{patient_id}/global/keypoints_hes_binary": len(kp1_binary) if kp1_binary else 0,
        f"{patient_id}/global/keypoints_cd30_binary": len(kp2_binary) if kp2_binary else 0,
        f"{patient_id}/global/keypoints_hes_total": len(kp1_combined),
        f"{patient_id}/global/keypoints_cd30_total": len(kp2_combined),
    })
    
    if desc1_combined is None or desc2_combined is None or len(kp1_combined) < 4 or len(kp2_combined) < 4:
        print("  ✗ Pas assez de points détectés pour la registration globale")
        wandb.log({f"{patient_id}/global/status": "failed_detection"})
        return None, lowres_cd30_np
    
    # Matching (AKAZE utilise des descripteurs binaires)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1_combined, desc2_combined)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # AKAZE produit généralement plus de matches de qualité, on peut être plus sélectif
    num_good_matches = min(len(matches), max(100, int(len(matches) * 0.25)))
    good_matches = matches[:num_good_matches]
    
    print(f"  Correspondances: {len(matches)} total, {len(good_matches)} sélectionnées")
    
    wandb.log({
        f"{patient_id}/global/total_matches": len(matches),
        f"{patient_id}/global/good_matches": len(good_matches),
    })
    
    if len(good_matches) < 4:
        print("  ✗ Pas assez de correspondances pour la registration globale")
        wandb.log({f"{patient_id}/global/status": "failed_matching"})
        return None, lowres_cd30_np
    
    # Extraire les points
    src_pts = np.float32([kp2_combined[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([kp1_combined[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    
    # RANSAC pour la transformation globale
    try:
        model_global, inliers = ransac(
            (src_pts, dst_pts),
            AffineTransform,
            min_samples=3,
            residual_threshold=8.0,  # Un peu plus permissif pour la lame entière
            max_trials=2000
        )
        
        if model_global is None or np.sum(inliers) < 4:
            print("  ✗ RANSAC échoué pour la registration globale")
            wandb.log({f"{patient_id}/global/status": "failed_ransac"})
            return None, lowres_cd30_np
        
        num_inliers = np.sum(inliers)
        inlier_ratio = num_inliers / len(good_matches)
        
        print(f"  ✓ Registration globale réussie: {num_inliers}/{len(good_matches)} inliers (ratio: {inlier_ratio:.3f})")
        
        # Logger les résultats
        wandb.log({
            f"{patient_id}/global/inliers": int(num_inliers),
            f"{patient_id}/global/inlier_ratio": inlier_ratio,
            f"{patient_id}/global/status": "success"
        })
        
        # Appliquer la transformation globale
        h, w = lowres_hes_np.shape[:2]
        aligned_cd30_global = cv2.warpAffine(
            lowres_cd30_np, 
            model_global.params[:2], 
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        
        # Visualisation de la registration globale avec les masques
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        
        # Première ligne : Images originales et alignées
        axes[0, 0].imshow(lowres_hes_np)
        axes[0, 0].set_title('HES (référence)', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(lowres_cd30_np)
        axes[0, 1].set_title('CD30 (originale)', fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(aligned_cd30_global)
        axes[0, 2].set_title(f'CD30 (alignée)\n{num_inliers} inliers, ratio: {inlier_ratio:.3f}', 
                           fontsize=12, fontweight='bold')
        axes[0, 2].axis('off')
        
        # Superposition pour validation
        overlay = cv2.addWeighted(lowres_hes_np, 0.5, aligned_cd30_global, 0.5, 0)
        axes[0, 3].imshow(overlay)
        axes[0, 3].set_title('Superposition HES+CD30', fontsize=12, fontweight='bold')
        axes[0, 3].axis('off')
        
        # Deuxième ligne : Masques utilisés pour la registration
        axes[1, 0].imshow(mask_hes, cmap='gray')
        axes[1, 0].set_title('Masque HES', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(mask_cd30, cmap='gray')
        axes[1, 1].set_title('Masque CD30', fontsize=12, fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(enhanced_mask_hes, cmap='gray')
        axes[1, 2].set_title('Masque HES amélioré', fontsize=12, fontweight='bold')
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(enhanced_mask_cd30, cmap='gray')
        axes[1, 3].set_title('Masque CD30 amélioré', fontsize=12, fontweight='bold')
        axes[1, 3].axis('off')
        
        plt.suptitle(f'Patient {patient_id}\nRegistration globale basée sur les masques de tissu', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        output_path = os.path.join(output_folder, f'{patient_id}_0_global_registration_masks.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  → Visualisation avec masques sauvegardée: {output_path}")
        
        wandb.log({
            f"{patient_id}/global/visualization_masks": wandb.Image(output_path)
        })
        
        return model_global, aligned_cd30_global
        
    except Exception as e:
        print(f"  ✗ Erreur lors de la registration globale: {e}")
        wandb.log({
            f"{patient_id}/global/status": "failed_error",
            f"{patient_id}/global/error": str(e)
        })
        return None, lowres_cd30_np


def calculate_adaptive_region_size(slide_width, slide_height, target_regions_per_axis=4):
    """
    Calcule la taille de région adaptée aux dimensions de la lame.
    
    Args:
        slide_width: Largeur de la lame en pixels (niveau 0)
        slide_height: Hauteur de la lame en pixels (niveau 0)  
        target_regions_per_axis: Nombre cible de régions par axe
        
    Returns:
        region_size: Taille optimale des régions en pixels
    """
    # Calculer la taille basée sur la plus petite dimension
    min_dimension = min(slide_width, slide_height)
    
    # Taille de région = dimension minimale / nombre de régions cibles
    base_region_size = min_dimension // target_regions_per_axis
    
    # Arrondir à un multiple de 1000 pour la simplicité
    region_size = ((base_region_size // 1000) + 1) * 1000
    
    # Limites de sécurité
    region_size = max(region_size, 6000)   # Minimum 6000px
    region_size = min(region_size, 20000)  # Maximum 20000px
    
    return region_size


def process_one_slide_pair_visualization(hes_path, cd30_path):
    """Traite une paire de slides et génère toutes les visualisations pour toutes les régions."""
    
    print(f"\n{'='*80}")
    print(f"Traitement de: {os.path.basename(hes_path)}")
    print(f"{'='*80}")
    
    slide_hes = openslide.OpenSlide(hes_path)
    slide_cd30 = openslide.OpenSlide(cd30_path)
    
    patient_id = os.path.splitext(os.path.basename(hes_path))[0].replace("_HES", "")
    
    # Obtenir les dimensions de la lame niveau 0
    w0_hes, h0_hes = slide_hes.level_dimensions[0]
    w0_cd30, h0_cd30 = slide_cd30.level_dimensions[0]
    
    # Calculer la taille de région adaptée
    adaptive_region_size = calculate_adaptive_region_size(w0_hes, h0_hes)
    print(f"\n[ADAPTATION] Dimensions lame HES: {w0_hes}x{h0_hes}")
    print(f"[ADAPTATION] Taille de région adaptée: {adaptive_region_size}px")
    
    # Logger les informations de la slide
    wandb.log({
        f"{patient_id}/slide_dimensions_hes": f"{slide_hes.level_dimensions[0]}",
        f"{patient_id}/slide_dimensions_cd30": f"{slide_cd30.level_dimensions[0]}",
        f"{patient_id}/adaptive_region_size": adaptive_region_size,
    })
    
    # Charger les images basse résolution
    print("\n[1/4] Chargement des images basse résolution...")
    w_lr_hes, h_lr_hes = slide_hes.level_dimensions[lowres_level]
    w_lr_cd30, h_lr_cd30 = slide_cd30.level_dimensions[lowres_level]
    
    lowres_hes = slide_hes.read_region((0, 0), lowres_level, (w_lr_hes, h_lr_hes)).convert("RGB")
    lowres_cd30 = slide_cd30.read_region((0, 0), lowres_level, (w_lr_cd30, h_lr_cd30)).convert("RGB")
    
    lowres_hes_np = np.array(lowres_hes)
    lowres_cd30_np = np.array(lowres_cd30)
    
    # ÉTAPE DE REGISTRATION GLOBALE
    print("\n[2/4] Registration globale de la lame entière...")
    global_transform, aligned_cd30_global = register_whole_slide(lowres_hes_np, lowres_cd30_np, patient_id)
    
    # Utiliser l'image alignée globalement pour la suite
    if global_transform is not None:
        print("  ✓ Utilisation de la transformation globale pour les régions")
        lowres_cd30_np = aligned_cd30_global
    else:
        print("  ⚠ Pas de transformation globale, utilisation de l'image originale")
    
    # Calculer le masque de tissu
    print("\n[3/4] Calcul du masque de tissu...")
    mask_hes = compute_tissue_mask(lowres_hes)
    
    # Trouver toutes les régions avec du tissu et effectuer la registration
    print("\n[4/4] Registration des régions avec tissu...")
    downsample_hes = int(slide_hes.level_downsamples[lowres_level])
    downsample_cd30 = int(slide_cd30.level_downsamples[lowres_level])
    
    # Utiliser la taille de région adaptée au lieu de la constante globale
    current_region_size = adaptive_region_size
    
    # Calculer le pas (stride) avec overlap
    stride = int(current_region_size * (1 - overlap_percent))
    print(f"Taille de région adaptée: {current_region_size}px")
    print(f"Stride: {stride}px (overlap: {overlap_percent*100}%)")
    print(f"Nombre estimé de régions: {(w0_hes//stride) * (h0_hes//stride)}")
    
    regions_processed = 0
    region_idx = 0
    regions_info = []
    
    for region_y in range(0, h0_hes, stride):
        for region_x in range(0, w0_hes, stride):
            if region_x + current_region_size > w0_hes or region_y + current_region_size > h0_hes:
                continue
            
            region_x_lr = int(region_x / downsample_hes)
            region_y_lr = int(region_y / downsample_hes)
            region_size_lr = int(current_region_size / downsample_hes)
            
            region_mask = mask_hes[region_y_lr:region_y_lr+region_size_lr, 
                                   region_x_lr:region_x_lr+region_size_lr]
            
            tissue_percent = np.mean(region_mask > 0)
            
            if region_mask.size == 0 or tissue_percent < 0.80:
                continue
            
            print(f"\n✓ Région {region_idx} trouvée à x={region_x}, y={region_y} (tissu: {tissue_percent:.1%})")
            
            # Logger les informations de la région
            wandb.log({
                f"{patient_id}/region_{region_idx}/position_x": region_x,
                f"{patient_id}/region_{region_idx}/position_y": region_y,
                f"{patient_id}/region_{region_idx}/tissue_percent": tissue_percent,
            })
            
            # Extraire les sous-régions
            region_hes_lr = lowres_hes_np[region_y_lr:region_y_lr+region_size_lr, 
                                          region_x_lr:region_x_lr+region_size_lr]
            
            region_x_cd30_lr = int(region_x / downsample_cd30)
            region_y_cd30_lr = int(region_y / downsample_cd30)
            region_size_cd30_lr = int(current_region_size / downsample_cd30)
            
            if region_x_cd30_lr + region_size_cd30_lr > w_lr_cd30 or region_y_cd30_lr + region_size_cd30_lr > h_lr_cd30:
                continue
            
            region_cd30_lr = lowres_cd30_np[region_y_cd30_lr:region_y_cd30_lr+region_size_cd30_lr, 
                                           region_x_cd30_lr:region_x_cd30_lr+region_size_cd30_lr]
            
            print(f">>> Registration de la région {region_idx}...")
            metrics = perform_region_registration(
                region_hes_lr, region_cd30_lr, patient_id, region_idx
            )
            
            # Enregistrer les métriques
            registration_metrics[patient_id][region_idx] = metrics
            
            # Stocker les informations de la région pour la visualisation
            regions_info.append({
                'idx': region_idx,
                'x_lr': region_x_lr,
                'y_lr': region_y_lr,
                'size_lr': region_size_lr,
                'metrics': metrics,
                'img_hes': region_hes_lr.copy(),
                'img_cd30_original': region_cd30_lr.copy()
            })
            
            if metrics['status'] == 'success':
                regions_processed += 1
                print(f"✓ Région {region_idx} traitée avec succès")
            else:
                print(f"✗ Alignement échoué pour région {region_idx}")
            
            region_idx += 1
    
    slide_hes.close()
    slide_cd30.close()
    
    # Appliquer la correction par cohérence spatiale
    if regions_info:
        print(f"\n[CORRECTION] Application de la cohérence spatiale pour {patient_id}...")
        regions_info = apply_spatial_coherence_correction(regions_info, patient_id)
        
        # Recompter les régions traitées après correction
        regions_processed = sum(1 for r in regions_info if r['metrics']['status'] in ['success', 'interpolated'])
    
    # Générer les images CD30 alignées pour chaque région
    if regions_info:
        print(f"\n[ALIGNEMENT] Génération des images alignées pour {patient_id}...")
        regions_data = []
        
        for region_info in regions_info:
            img_hes = region_info['img_hes']
            img_cd30_original = region_info['img_cd30_original']
            transform = region_info['metrics']['transform']
            
            # Appliquer la transformation si elle existe
            if transform is not None:
                h, w = img_hes.shape[:2]
                img_cd30_aligned = cv2.warpAffine(
                    img_cd30_original,
                    transform.params[:2],
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255)
                )
            else:
                # Si pas de transformation, utiliser l'image originale
                img_cd30_aligned = img_cd30_original.copy()
            
            regions_data.append({
                'idx': region_info['idx'],
                'img_hes': img_hes,
                'img_cd30_original': img_cd30_original,
                'img_cd30_aligned': img_cd30_aligned,
                'metrics': region_info['metrics']
            })
    
    # Générer la figure de vue d'ensemble
    if regions_info:
        print(f"\n[VISUALISATION] Génération de la vue d'ensemble pour {patient_id}...")
        generate_patient_overview_figure(lowres_hes_np, patient_id, regions_info)
    
    # Générer la figure triptyque de toutes les régions
    if regions_data:
        print(f"\n[VISUALISATION] Génération du triptyque des régions pour {patient_id}...")
        generate_regions_triptych_figure(patient_id, regions_data)
    
    # Logger le résumé du patient
    wandb.log({
        f"{patient_id}/total_regions_found": region_idx,
        f"{patient_id}/regions_processed": regions_processed,
        f"{patient_id}/success_rate": regions_processed / region_idx if region_idx > 0 else 0,
    })
    
    print(f"\n{'='*80}")
    if regions_processed > 0:
        print(f"✓ {regions_processed} région(s) traitée(s) avec succès pour {patient_id}")
    else:
        print(f"✗ Aucune région n'a pu être traitée pour {patient_id}")
    print(f"{'='*80}")
    
    return regions_processed


# ============================================================================
# EXÉCUTION PRINCIPALE
# ============================================================================

print(f"\n{'='*80}")
print("VISUALISATION DU PROCESSUS AKAZE + RANSAC + COHÉRENCE SPATIALE")
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

# Traiter toutes les paires de patients
if pairs:
    total_patients = len(pairs)
    patients_with_regions = 0
    total_regions = 0
    
    print(f"\nTraitement de {total_patients} patient(s)...")
    
    for idx, (patient_id, paths) in enumerate(pairs.items(), 1):
        print(f"\n[Patient {idx}/{total_patients}] Traitement de: {patient_id}")
        regions_count = process_one_slide_pair_visualization(paths['hes'], paths['cd30'])
        
        if regions_count > 0:
            patients_with_regions += 1
            total_regions += regions_count
    
    # Logger le résumé final
    wandb.log({
        "summary/total_patients": total_patients,
        "summary/patients_with_regions": patients_with_regions,
        "summary/total_regions_processed": total_regions,
        "summary/avg_regions_per_patient": total_regions / total_patients if total_patients > 0 else 0,
    })
    
    print(f"\n{'='*80}")
    print(f"✓ TRAITEMENT TERMINÉ")
    print(f"{'='*80}")
    print(f"Patients traités: {total_patients}")
    print(f"Patients avec régions réussies: {patients_with_regions}")
    print(f"Total de régions traitées: {total_regions}")
    print(f"Les visualisations sont sauvegardées dans: {output_folder}")
    print(f"{'='*80}")
    
    # Finir la session wandb
    wandb.finish()
else:
    print("\n✗ Aucune paire de slides trouvée")
    wandb.finish()
