import cv2
import matplotlib.pyplot as plt
import os
import re
from pathlib import Path

# Dossiers source
cd30_base = Path("/silver/ube/extract/CD30/AHL001")
hes_base = Path("/silver/ube/extract/HES/AHL001")

# Nombre d'images à afficher
n_pairs = 20

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

# Récupérer tous les fichiers CD30 avec leur région et créer un dictionnaire (region, coords) -> fichier
cd30_dict = {}
cd30_count = 0
for region_dir in cd30_base.glob("region_*"):
    if region_dir.is_dir():
        for f in region_dir.glob("*.jpg"):
            coords = extract_coords(f.name)
            if coords:
                region_name = get_region_name(region_dir)
                cd30_dict[(region_name, coords)] = f
                cd30_count += 1

# Récupérer tous les fichiers HES avec leur région et créer un dictionnaire (region, coords) -> fichier
hes_dict = {}
hes_count = 0
for region_dir in hes_base.glob("region_*"):
    if region_dir.is_dir():
        for f in region_dir.glob("*.jpg"):
            coords = extract_coords(f.name)
            if coords:
                region_name = get_region_name(region_dir)
                hes_dict[(region_name, coords)] = f
                hes_count += 1

# Trouver les clés communes (paires qui existent dans les deux)
common_keys = sorted(set(cd30_dict.keys()) & set(hes_dict.keys()))

print(f"Found {cd30_count} CD30 images")
print(f"Found {hes_count} HES images")
print(f"Found {len(common_keys)} matching pairs")
print(f"Will display {min(n_pairs, len(common_keys))} pairs")

# Afficher les paires
for i, key in enumerate(common_keys[:n_pairs]):
    region_name, coords = key
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Charger et afficher CD30
    cd30_path = cd30_dict[key]
    img_cd30 = cv2.imread(str(cd30_path))
    img_cd30 = cv2.cvtColor(img_cd30, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_cd30)
    axes[0].set_title(f'CD30 #{i+1}\n{cd30_path.parent.name}/{cd30_path.name}', fontsize=10)
    axes[0].axis('off')
    
    # Charger et afficher HES
    hes_path = hes_dict[key]
    img_hes = cv2.imread(str(hes_path))
    img_hes = cv2.cvtColor(img_hes, cv2.COLOR_BGR2RGB)
    axes[1].imshow(img_hes)
    axes[1].set_title(f'HES #{i+1}\n{hes_path.parent.name}/{hes_path.name}', fontsize=10)
    axes[1].axis('off')
    
    plt.suptitle(f'Pair {i+1}/{min(n_pairs, len(common_keys))} - Region: {region_name} - Coords: x={coords[0]}, y={coords[1]}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

print("Visualization complete!")
