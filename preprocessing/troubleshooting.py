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

# Login to Weights & Biases
wandb.login(key="ab67e0f4c27fad7a0d47405f84a8a4deb80056ba")

# Configuration
input_folder = "/gold/data_feasibility"
output_folder = "./visualizations_ORB"
os.makedirs(output_folder, exist_ok=True)

# Dictionnaire global pour stocker les métriques de registration
registration_metrics = defaultdict(lambda: defaultdict(dict))

patch_size = 2000
region_size = 12000
overlap_percent = 0.20  # 20% de chevauchement entre les régions
lowres_level = 2

# Initialiser wandb
wandb.init(
    project="ia2hl-preprocessing",
    name="orb-registration-visualization",
    config={
        "patch_size": patch_size,
        "region_size": region_size,
        "overlap_percent": overlap_percent,
        "lowres_level": lowres_level,
        "input_folder": input_folder,
        "output_folder": output_folder
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
        # Enregistrer les métriques d'échec
        registration_metrics[patient_id][region_idx] = {
            'inliers': 0,
            'total_matches': 0,
            'ratio': 0.0,
            'status': 'failed_detection'
        }
        return None, None, None, False
    
    # Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    num_good_matches = min(len(matches), max(50, int(len(matches) * 0.15)))
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
        # Enregistrer les métriques d'échec
        registration_metrics[patient_id][region_idx] = {
            'inliers': 0,
            'total_matches': len(matches),
            'ratio': 0.0,
            'status': 'failed_matching'
        }
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
            # Enregistrer les métriques d'échec
            registration_metrics[patient_id][region_idx] = {
                'inliers': int(num_inliers_found),
                'total_matches': len(good_matches),
                'ratio': 0.0,
                'status': f'failed_insufficient_inliers (found {num_inliers_found}, required {MIN_INLIERS})'
            }
            return None, None, None, False
        
        num_inliers = np.sum(inliers)
        num_outliers = len(good_matches) - num_inliers
        inlier_ratio = num_inliers / len(good_matches)
        
        print(f"✓ {num_inliers} inliers sur {len(good_matches)} matches (seuil: {MIN_INLIERS})")
        
        # Logger les résultats RANSAC
        wandb.log({
            f"{patient_id}/region_{region_idx}/inliers": int(num_inliers),
            f"{patient_id}/region_{region_idx}/outliers": int(num_outliers),
            f"{patient_id}/region_{region_idx}/inlier_ratio": inlier_ratio,
        })
        
        # Enregistrer les métriques de succès
        registration_metrics[patient_id][region_idx] = {
            'inliers': int(num_inliers),
            'total_matches': len(good_matches),
            'ratio': float(inlier_ratio),
            'status': 'success'
        }
        
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
        
        # Logger l'image de visualisation
        wandb.log({
            f"{patient_id}/region_{region_idx}/visualization": wandb.Image(output_path1),
            f"{patient_id}/region_{region_idx}/status": "success"
        })
        
        return model, aligned_img, inliers, True
        
    except Exception as e:
        print(f"Erreur: {e}")
        wandb.log({
            f"{patient_id}/region_{region_idx}/status": "failed_error",
            f"{patient_id}/region_{region_idx}/error": str(e)
        })
        # Enregistrer les métriques d'erreur
        registration_metrics[patient_id][region_idx] = {
            'inliers': 0,
            'total_matches': 0,
            'ratio': 0.0,
            'status': 'failed_error',
            'error': str(e)
        }
        return None, None, None, False



def generate_registration_csv(output_folder):
    """
    Génère un CSV avec les métriques de registration.
    Format: Lignes = patients_type (HES/CD30), Colonnes = régions, Valeurs = ratio inliers/total
    """
    csv_path = os.path.join(output_folder, "registration_metrics.csv")
    
    # Créer une liste pour stocker les données
    rows = []
    
    # Pour chaque patient
    for patient_id in sorted(registration_metrics.keys()):
        patient_data = registration_metrics[patient_id]
        
        # Créer une ligne pour les ratios
        row = {'Patient_Type': f"{patient_id}_HES/CD30"}
        
        # Pour chaque région
        for region_idx in sorted(patient_data.keys()):
            metrics = patient_data[region_idx]
            ratio = metrics['ratio']
            status = metrics['status']
            
            # Formater la valeur: ratio ou indication d'échec
            if status == 'success':
                row[f"Region_{region_idx}"] = f"{ratio:.3f}"
            else:
                row[f"Region_{region_idx}"] = f"0.000 ({status})"
        
        rows.append(row)
    
    # Créer le DataFrame
    df = pd.DataFrame(rows)
    
    # Réorganiser les colonnes pour avoir Patient_Type en premier, puis les régions triées
    region_cols = sorted([col for col in df.columns if col.startswith('Region_')], 
                        key=lambda x: int(x.split('_')[1]))
    df = df[['Patient_Type'] + region_cols]
    
    # Sauvegarder le CSV
    df.to_csv(csv_path, index=False)
    print(f"\n✓ CSV de métriques sauvegardé: {csv_path}")
    
    # Créer aussi un CSV détaillé avec inliers et total
    detailed_csv_path = os.path.join(output_folder, "registration_metrics_detailed.csv")
    detailed_rows = []
    
    for patient_id in sorted(registration_metrics.keys()):
        patient_data = registration_metrics[patient_id]
        
        for region_idx in sorted(patient_data.keys()):
            metrics = patient_data[region_idx]
            detailed_rows.append({
                'Patient': patient_id,
                'Type': 'HES/CD30',
                'Region': region_idx,
                'Inliers': metrics['inliers'],
                'Total_Matches': metrics['total_matches'],
                'Ratio': metrics['ratio'],
                'Status': metrics['status']
            })
    
    df_detailed = pd.DataFrame(detailed_rows)
    df_detailed.to_csv(detailed_csv_path, index=False)
    print(f"✓ CSV détaillé sauvegardé: {detailed_csv_path}")
    
    # Logger le CSV dans wandb
    wandb.log({
        "registration_metrics_table": wandb.Table(dataframe=df),
        "registration_metrics_detailed": wandb.Table(dataframe=df_detailed)
    })
    
    return csv_path, detailed_csv_path


def register_whole_slide(lowres_hes_np, lowres_cd30_np, patient_id):
    """
    Effectue une registration globale de la lame entière à basse résolution.
    Retourne la transformation globale et l'image CD30 alignée.
    """
    print("\n[REGISTRATION GLOBALE] Alignement de la lame entière...")
    
    # Conversion en niveaux de gris
    gray_hes = cv2.cvtColor(lowres_hes_np, cv2.COLOR_RGB2GRAY)
    gray_cd30 = cv2.cvtColor(lowres_cd30_np, cv2.COLOR_RGB2GRAY)
    
    # Détection ORB sur la lame entière
    orb = cv2.ORB_create(nfeatures=8000)  # Plus de features pour la lame entière
    kp1, desc1 = orb.detectAndCompute(gray_hes, None)
    kp2, desc2 = orb.detectAndCompute(gray_cd30, None)
    
    print(f"  Points détectés - HES: {len(kp1) if kp1 else 0}, CD30: {len(kp2) if kp2 else 0}")
    
    # Logger les keypoints globaux
    wandb.log({
        f"{patient_id}/global/keypoints_hes": len(kp1) if kp1 else 0,
        f"{patient_id}/global/keypoints_cd30": len(kp2) if kp2 else 0,
    })
    
    if desc1 is None or desc2 is None or len(kp1) < 4 or len(kp2) < 4:
        print("  ✗ Pas assez de points détectés pour la registration globale")
        wandb.log({f"{patient_id}/global/status": "failed_detection"})
        return None, lowres_cd30_np
    
    # Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    num_good_matches = min(len(matches), max(100, int(len(matches) * 0.20)))
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
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    
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
        
        # Visualisation de la registration globale
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        
        axes[0].imshow(lowres_hes_np)
        axes[0].set_title('HES (référence)', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(lowres_cd30_np)
        axes[1].set_title('CD30 (originale)', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        axes[2].imshow(aligned_cd30_global)
        axes[2].set_title(f'CD30 (alignée globalement)\n{num_inliers} inliers, ratio: {inlier_ratio:.3f}', 
                         fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(f'Patient {patient_id}\nRegistration globale de la lame entière', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        output_path = os.path.join(output_folder, f'{patient_id}_0_global_registration.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  → Visualisation sauvegardée: {output_path}")
        
        wandb.log({
            f"{patient_id}/global/visualization": wandb.Image(output_path)
        })
        
        return model_global, aligned_cd30_global
        
    except Exception as e:
        print(f"  ✗ Erreur lors de la registration globale: {e}")
        wandb.log({
            f"{patient_id}/global/status": "failed_error",
            f"{patient_id}/global/error": str(e)
        })
        return None, lowres_cd30_np


def process_one_slide_pair_visualization(hes_path, cd30_path):
    """Traite une paire de slides et génère toutes les visualisations pour toutes les régions."""
    
    print(f"\n{'='*80}")
    print(f"Traitement de: {os.path.basename(hes_path)}")
    print(f"{'='*80}")
    
    slide_hes = openslide.OpenSlide(hes_path)
    slide_cd30 = openslide.OpenSlide(cd30_path)
    
    patient_id = os.path.splitext(os.path.basename(hes_path))[0].replace("_HES", "")
    
    # Logger les informations de la slide
    wandb.log({
        f"{patient_id}/slide_dimensions_hes": f"{slide_hes.level_dimensions[0]}",
        f"{patient_id}/slide_dimensions_cd30": f"{slide_cd30.level_dimensions[0]}",
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
    
    # Trouver toutes les régions avec du tissu
    print("\n[4/4] Recherche de toutes les régions avec tissu...")
    w0_hes, h0_hes = slide_hes.level_dimensions[0]
    downsample_hes = int(slide_hes.level_downsamples[lowres_level])
    downsample_cd30 = int(slide_cd30.level_downsamples[lowres_level])
    
    # Calculer le pas (stride) avec overlap
    stride = int(region_size * (1 - overlap_percent))
    print(f"Taille de région: {region_size}px, Stride: {stride}px (overlap: {overlap_percent*100}%)")
    
    regions_processed = 0
    region_idx = 0
    
    for region_y in range(0, h0_hes, stride):
        for region_x in range(0, w0_hes, stride):
            if region_x + region_size > w0_hes or region_y + region_size > h0_hes:
                continue
            
            region_x_lr = int(region_x / downsample_hes)
            region_y_lr = int(region_y / downsample_hes)
            region_size_lr = int(region_size / downsample_hes)
            
            region_mask = mask_hes[region_y_lr:region_y_lr+region_size_lr, 
                                   region_x_lr:region_x_lr+region_size_lr]
            
            tissue_percent = np.mean(region_mask > 0)
            
            if region_mask.size == 0 or tissue_percent < 0.60:
                continue
            
            print(f"\n✓ Région {region_idx} trouvée à x={region_x}, y={region_y}")
            
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
            region_size_cd30_lr = int(region_size / downsample_cd30)
            
            if region_x_cd30_lr + region_size_cd30_lr > w_lr_cd30 or region_y_cd30_lr + region_size_cd30_lr > h_lr_cd30:
                continue
            
            region_cd30_lr = lowres_cd30_np[region_y_cd30_lr:region_y_cd30_lr+region_size_cd30_lr, 
                                           region_x_cd30_lr:region_x_cd30_lr+region_size_cd30_lr]
            
            print(f">>> Génération de la visualisation pour région {region_idx}: Détection ORB et appariement")
            transformation, aligned_img, inliers, success = visualize_orb_detection_and_matching(
                region_hes_lr, region_cd30_lr, patient_id, region_idx
            )
            
            if success:
                regions_processed += 1
                print(f"✓ Région {region_idx} traitée avec succès")
            else:
                print(f"✗ Alignement échoué pour région {region_idx}")
            
            region_idx += 1
    
    slide_hes.close()
    slide_cd30.close()
    
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
    
    # Générer les CSV de métriques
    print(f"\n{'='*80}")
    print("GÉNÉRATION DES MÉTRIQUES CSV")
    print(f"{'='*80}")
    csv_path, detailed_csv_path = generate_registration_csv(output_folder)
    
    print(f"\n{'='*80}")
    print(f"✓ TRAITEMENT TERMINÉ")
    print(f"{'='*80}")
    print(f"Patients traités: {total_patients}")
    print(f"Patients avec régions réussies: {patients_with_regions}")
    print(f"Total de régions traitées: {total_regions}")
    print(f"Les visualisations sont sauvegardées dans: {output_folder}")
    print(f"Métriques CSV: {csv_path}")
    print(f"Métriques détaillées: {detailed_csv_path}")
    print(f"{'='*80}")
    
    # Finir la session wandb
    wandb.finish()
else:
    print("\n✗ Aucune paire de slides trouvée")
    wandb.finish()
