import cv2
import matplotlib.pyplot as plt
import os
import re
from pathlib import Path
import numpy as np

# Configuration
patient_ids = ["AHL001", "AHL002", "AHL004", "AHL006", "AHL011"]
base_path = Path("/silver/ube/extract")
output_file = "pair_matching_overview.jpg"

# Nombre de paires à afficher par patient (réparties sur différentes régions)
n_pairs_per_patient = 4

def extract_coords(filename):
    """Extrait les coordonnées x, y du nom de fichier"""
    # Exemple: patch_x0_y0.jpg -> (0, 0)
    match = re.search(r'patch_x(\d+)_y(\d+)', filename)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None

def get_region_name(region_dir):
    """Extrait le nom de la région depuis le chemin"""
    return region_dir.name

def process_patient(patient_id):
    """Traite un patient et retourne les paires d'images sélectionnées."""
    print(f"\nTraitement de {patient_id}...")
    
    cd30_base = base_path / "CD30" / patient_id
    hes_base = base_path / "HES" / patient_id
    
    if not cd30_base.exists() or not hes_base.exists():
        print(f"  ⚠ Dossiers manquants pour {patient_id}")
        return []
    
    # Récupérer tous les fichiers CD30 avec leur région
    cd30_dict = {}
    for region_dir in cd30_base.glob("region_*"):
        if region_dir.is_dir():
            for f in region_dir.glob("*.jpg"):
                coords = extract_coords(f.name)
                if coords:
                    region_name = get_region_name(region_dir)
                    cd30_dict[(region_name, coords)] = f
    
    # Récupérer tous les fichiers HES avec leur région
    hes_dict = {}
    for region_dir in hes_base.glob("region_*"):
        if region_dir.is_dir():
            for f in region_dir.glob("*.jpg"):
                coords = extract_coords(f.name)
                if coords:
                    region_name = get_region_name(region_dir)
                    hes_dict[(region_name, coords)] = f
    
    # Trouver les clés communes (paires qui existent dans les deux)
    common_keys = sorted(set(cd30_dict.keys()) & set(hes_dict.keys()))
    
    print(f"  Trouvé {len(hes_dict)} patches HES, {len(cd30_dict)} patches CD30")
    print(f"  {len(common_keys)} paires correspondantes")
    
    # Sélectionner des paires représentatives de différentes régions
    # Grouper par région
    regions = {}
    for key in common_keys:
        region_name = key[0]
        if region_name not in regions:
            regions[region_name] = []
        regions[region_name].append(key)
    
    # Sélectionner une paire par région (jusqu'à n_pairs_per_patient)
    selected_keys = []
    region_list = sorted(regions.keys())
    
    # Distribuer les paires équitablement sur les régions
    pairs_per_region = max(1, n_pairs_per_patient // len(region_list)) if len(region_list) > 0 else 1
    
    for region_name in region_list:
        if len(selected_keys) >= n_pairs_per_patient:
            break
        # Prendre les premières paires de cette région
        selected_keys.extend(regions[region_name][:pairs_per_region])
    
    # Si on n'a pas assez, compléter avec d'autres paires
    if len(selected_keys) < n_pairs_per_patient:
        for key in common_keys:
            if key not in selected_keys:
                selected_keys.append(key)
                if len(selected_keys) >= n_pairs_per_patient:
                    break
    
    # Charger les images
    pairs = []
    for key in selected_keys[:n_pairs_per_patient]:
        region_name, coords = key
        
        cd30_path = cd30_dict[key]
        hes_path = hes_dict[key]
        
        img_cd30 = cv2.imread(str(cd30_path))
        img_cd30 = cv2.cvtColor(img_cd30, cv2.COLOR_BGR2RGB)
        
        img_hes = cv2.imread(str(hes_path))
        img_hes = cv2.cvtColor(img_hes, cv2.COLOR_BGR2RGB)
        
        pairs.append({
            'hes': img_hes,
            'cd30': img_cd30,
            'region': region_name,
            'coords': coords,
            'patient_id': patient_id
        })
    
    print(f"  ✓ {len(pairs)} paires sélectionnées de {len(regions)} régions")
    return pairs


# EXÉCUTION PRINCIPALE
print("="*80)
print("GÉNÉRATION DE LA FIGURE GLOBALE DE CORRESPONDANCE DES PAIRES")
print("="*80)

all_pairs = []
for patient_id in patient_ids:
    try:
        pairs = process_patient(patient_id)
        all_pairs.extend(pairs)
    except Exception as e:
        print(f"  ⚠ Erreur pour {patient_id}: {e}")
        import traceback
        traceback.print_exc()
        continue

# Créer la grande figure
print("\n" + "="*80)
print("CRÉATION DE LA FIGURE FINALE")
print("="*80)

n_total_pairs = len(all_pairs)
n_cols = 2  # HES et CD30 côte à côte
n_rows = n_total_pairs

fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))

# S'assurer que axes est un tableau 2D même avec une seule paire
if n_rows == 1:
    axes = axes.reshape(1, -1)

for idx, pair in enumerate(all_pairs):
    # HES
    axes[idx, 0].imshow(pair['hes'])
    axes[idx, 0].set_title(f"{pair['patient_id']} - H&E\n{pair['region']}", 
                           fontsize=11, fontweight='bold')
    axes[idx, 0].axis('off')
    
    # CD30
    axes[idx, 1].imshow(pair['cd30'])
    axes[idx, 1].set_title(f"{pair['patient_id']} - CD30\n{pair['region']}", 
                           fontsize=11, fontweight='bold')
    axes[idx, 1].axis('off')
    
    # Ajouter un cadre coloré pour identifier le patient
    patient_colors = {
        'AHL001': '#e74c3c',
        'AHL002': '#3498db', 
        'AHL004': '#2ecc71',
        'AHL006': '#f39c12',
        'AHL011': '#9b59b6'
    }
    color = patient_colors.get(pair['patient_id'], '#95a5a6')
    
    for ax in [axes[idx, 0], axes[idx, 1]]:
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(4)
            spine.set_visible(True)

plt.suptitle('Correspondance des paires H&E / CD30 par sous-région', 
             fontsize=16, fontweight='bold', y=0.998)
plt.tight_layout()

# Sauvegarder en haute qualité
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='jpg')
print(f"\n✓ Figure sauvegardée: {output_file}")
print(f"  Format: JPEG haute qualité (300 DPI)")
print(f"  Nombre total de paires: {n_total_pairs}")
print(f"  Patients inclus: {len(patient_ids)}")

print("\n✓ Visualisation terminée!")
