import cv2
import matplotlib.pyplot as plt
import os
import re
from pathlib import Path

# Dossiers source
cd30_dir = Path("/home/azureuser/CD30/AHL001_CD30")
hes_dir = Path("/home/azureuser/HES/AHL001_HES")

# Nombre d'images à afficher
n_pairs = 20

def extract_coords(filename):
    """Extrait les coordonnées x, y du nom de fichier"""
    # Exemple: AHL001_CD30_x56832_y32768.png -> (56832, 32768)
    match = re.search(r'_x(\d+)_y(\d+)', filename)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None

# Récupérer tous les fichiers CD30 et créer un dictionnaire coords -> fichier
cd30_files = list(cd30_dir.glob("*.png"))
cd30_dict = {}
for f in cd30_files:
    coords = extract_coords(f.name)
    if coords:
        cd30_dict[coords] = f

# Récupérer tous les fichiers HES et créer un dictionnaire coords -> fichier
hes_files = list(hes_dir.glob("*.png"))
hes_dict = {}
for f in hes_files:
    coords = extract_coords(f.name)
    if coords:
        hes_dict[coords] = f

# Trouver les coordonnées communes (paires qui existent dans les deux)
common_coords = sorted(set(cd30_dict.keys()) & set(hes_dict.keys()))

print(f"Found {len(cd30_files)} CD30 images")
print(f"Found {len(hes_files)} HES images")
print(f"Found {len(common_coords)} matching pairs")
print(f"Will display {min(n_pairs, len(common_coords))} pairs")

# Afficher les paires
for i, coords in enumerate(common_coords[:n_pairs]):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Charger et afficher CD30
    cd30_path = cd30_dict[coords]
    img_cd30 = cv2.imread(str(cd30_path))
    img_cd30 = cv2.cvtColor(img_cd30, cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_cd30)
    axes[0].set_title(f'CD30 #{i+1}\n{cd30_path.name}', fontsize=10)
    axes[0].axis('off')
    
    # Charger et afficher HES
    hes_path = hes_dict[coords]
    img_hes = cv2.imread(str(hes_path))
    img_hes = cv2.cvtColor(img_hes, cv2.COLOR_BGR2RGB)
    axes[1].imshow(img_hes)
    axes[1].set_title(f'HES #{i+1}\n{hes_path.name}', fontsize=10)
    axes[1].axis('off')
    
    plt.suptitle(f'Pair {i+1}/{min(n_pairs, len(common_coords))} - Coords: x={coords[0]}, y={coords[1]}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

print("Visualization complete!")
