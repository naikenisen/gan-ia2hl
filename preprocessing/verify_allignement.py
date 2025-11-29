import cv2
import matplotlib.pyplot as plt
import os
import re
from pathlib import Path

# Dossiers source
cd30_dir = Path("results/CD30")
hes_dir = Path("results/HES")

# Nombre d'images à afficher
n_pairs = 20

def extract_coords(filename):
    """Extrait les coordonnées x, y du nom de fichier"""
    # Exemple: AHL001_CD30_x56832_y32768.jpg -> (56832, 32768)
    match = re.search(r'_x(\d+)_y(\d+)', filename)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


# Récupérer tous les fichiers CD30 (récursif dans les sous-dossiers)
cd30_files = list(cd30_dir.rglob("*.jpg"))
cd30_dict = {}
for f in cd30_files:
    coords = extract_coords(f.name)
    if coords:
        cd30_dict[coords] = f

# Récupérer tous les fichiers HES (récursif dans les sous-dossiers)
hes_files = list(hes_dir.rglob("*.jpg"))
hes_dict = {}
for f in hes_files:
    coords = extract_coords(f.name)
    if coords:
        hes_dict[coords] = f


# Sélectionner une seule paire par sous-dossier/région (pour chaque région HES et CD30)
from collections import defaultdict

# Associer chaque fichier à son sous-dossier parent (région)
def get_region(file_path, root_dir):
    rel = file_path.relative_to(root_dir)
    if len(rel.parts) > 1:
        return rel.parts[0]
    return None

# Regrouper les paires par région (sous-dossier HES)
region_pairs = defaultdict(list)
for coords in set(cd30_dict.keys()) & set(hes_dict.keys()):
    hes_path = hes_dict[coords]
    cd30_path = cd30_dict[coords]
    region = get_region(hes_path, hes_dir)
    if region is not None:
        region_pairs[region].append((coords, hes_path, cd30_path))

# Pour chaque région, tirer une seule paire au hasard (ou la première)
import random
selected_pairs = []
for region, pairs in region_pairs.items():
    if pairs:
        selected_pairs.append(random.choice(pairs))

print(f"Found {len(cd30_files)} CD30 images")
print(f"Found {len(hes_files)} HES images")
print(f"Found {len(selected_pairs)} region pairs (1 par région)")
print(f"Will display {min(n_pairs, len(selected_pairs))} pairs")

# Créer une planche (grid) avec 1 paire par sous-dossier/région et enregistrer la figure
import math

num_pairs = min(n_pairs, len(selected_pairs))
fig, axes = plt.subplots(num_pairs, 2, figsize=(15, 5 * num_pairs))
if num_pairs == 1:
    axes = [axes]  # pour le cas d'une seule paire

for i, (coords, hes_path, cd30_path) in enumerate(selected_pairs[:num_pairs]):
    # Charger et afficher CD30
    img_cd30 = cv2.imread(str(cd30_path))
    img_cd30 = cv2.cvtColor(img_cd30, cv2.COLOR_BGR2RGB)
    axes[i][0].imshow(img_cd30)
    axes[i][0].set_title(f'CD30 #{i+1}\n{cd30_path.name}', fontsize=10)
    axes[i][0].axis('off')

    # Charger et afficher HES
    img_hes = cv2.imread(str(hes_path))
    img_hes = cv2.cvtColor(img_hes, cv2.COLOR_BGR2RGB)
    axes[i][1].imshow(img_hes)
    axes[i][1].set_title(f'HES #{i+1}\n{hes_path.name}', fontsize=10)
    axes[i][1].axis('off')

    # Ajouter le titre de la paire
    axes[i][0].set_ylabel(f'Region: {get_region(hes_path, hes_dir)}\nCoords: x={coords[0]}, y={coords[1]}', fontsize=12, fontweight='bold')

plt.suptitle(f'Planche de {num_pairs} paires HES/CD30 (1 par région)', fontsize=18, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
output_path = "planche_paires_regions.png"
plt.savefig(output_path, dpi=200)
plt.close()
print(f"Planche enregistrée sous {output_path}")
print("Visualization complete!")
