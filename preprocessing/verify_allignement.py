import cv2
import matplotlib.pyplot as plt
import os
import re
from pathlib import Path
from collections import defaultdict
import random
import math

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

# Associer chaque fichier à son dossier patient
def get_patient(file_path, root_dir):
    rel = file_path.relative_to(root_dir)
    if len(rel.parts) > 1:
        return rel.parts[0]
    return None

# Regrouper les paires par patient
patient_pairs = defaultdict(list)
for coords in set(cd30_dict.keys()) & set(hes_dict.keys()):
    hes_path = hes_dict[coords]
    cd30_path = cd30_dict[coords]
    patient = get_patient(hes_path, hes_dir)
    if patient is not None:
        patient_pairs[patient].append((coords, hes_path, cd30_path))

# Pour chaque patient, tirer plusieurs paires au hasard pour atteindre n_pairs
selected_pairs = []
all_pairs = []
for patient, pairs in patient_pairs.items():
    all_pairs.extend(pairs)

# Sélectionner aléatoirement n_pairs parmi toutes les paires disponibles
if all_pairs:
    selected_pairs = random.sample(all_pairs, min(n_pairs, len(all_pairs)))
num_pairs = min(n_pairs, len(selected_pairs))
fig, axes = plt.subplots(num_pairs, 2, figsize=(15, 5 * num_pairs))

for i, (coords, hes_path, cd30_path) in enumerate(selected_pairs[:num_pairs]):
    img_cd30 = cv2.imread(str(cd30_path))
    img_cd30 = cv2.cvtColor(img_cd30, cv2.COLOR_BGR2RGB)
    axes[i][0].imshow(img_cd30)
    axes[i][0].set_title(f'CD30 #{i+1}\n{cd30_path.name}', fontsize=10)
    axes[i][0].axis('off')
    img_hes = cv2.imread(str(hes_path))
    img_hes = cv2.cvtColor(img_hes, cv2.COLOR_BGR2RGB)
    axes[i][1].imshow(img_hes)
    axes[i][1].set_title(f'HES #{i+1}\n{hes_path.name}', fontsize=10)
    axes[i][1].axis('off')
    patient = get_patient(hes_path, hes_dir)
    axes[i][0].set_ylabel(f'Patient: {patient}\nCoords: x={coords[0]}, y={coords[1]}', fontsize=12, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig("/silver/ube/visualization/planche_paires_patches.png", dpi=200)
plt.close()