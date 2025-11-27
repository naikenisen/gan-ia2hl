import os
import math
import openslide
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

folder = "/gold/data_feasibility"
all_svs_files = [f for f in os.listdir(folder) if f.lower().endswith(".svs")]

# ============================================================================
# CONFIGURATION : Choisissez les lames à visualiser
# ============================================================================
# Option 1: Laisser vide pour traiter TOUTES les lames
# svs_files = all_svs_files

# Option 2: Spécifier les noms de fichiers (avec ou sans extension)
svs_files = [
    "AHL004_HES.svs",
    "AHL004_CD30.svs",
    "AHL006_HES.svs",
    "AHL006_CD30.svs"
]

# Option 3: Filtrer par type (HES ou CD30)
# svs_files = [f for f in all_svs_files if "_HES" in f]
# svs_files = [f for f in all_svs_files if "_CD30" in f]

# Option 4: Sélectionner les N premières lames
# svs_files = all_svs_files[:5]

# Si aucune sélection spécifique, afficher la liste disponible
if not svs_files:
    print("\n" + "="*80)
    print("LAMES DISPONIBLES:")
    print("="*80)
    for i, f in enumerate(all_svs_files, 1):
        print(f"{i:2d}. {f}")
    print("\n⚠️  Aucune lame sélectionnée!")
    print("Modifiez la variable 'svs_files' dans le code pour choisir vos lames.")
    print("="*80 + "\n")
    exit()

# Vérifier que les fichiers sélectionnés existent
svs_files_valid = []
for f in svs_files:
    # Ajouter .svs si pas présent
    if not f.endswith('.svs'):
        f = f + '.svs'
    
    if f in all_svs_files:
        svs_files_valid.append(f)
    else:
        print(f"⚠️  Fichier non trouvé: {f}")

svs_files = svs_files_valid

if not svs_files:
    print("\n✗ Aucune lame valide à traiter!")
    exit()

print(f"\n✓ {len(svs_files)} lame(s) sélectionnée(s) pour visualisation\n")

level = 2

def compute_tissue_mask(img_rgb):
    img_np = np.array(img_rgb)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

n = len(svs_files)
cols = 4
rows = 3 * math.ceil(n / cols)  # 3 rows per slide

plt.figure(figsize=(5 * cols, 5 * rows))

for idx, filename in enumerate(tqdm(svs_files, desc="Processing slides")):
    path = os.path.join(folder, filename)
    slide = openslide.OpenSlide(path)

    # low-res
    w, h = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, (w, h)).convert("RGB")

    # mask
    mask = compute_tissue_mask(img)
    img_np = np.array(img)
    img_masked = img_np.copy()
    img_masked[mask == 0] = 255

    # compute base row for this slide
    base_row = (idx // cols) * 3      # 3 rows per slide
    col = idx % cols

    # --- Plot low-res ---
    plt.subplot(rows, cols, base_row * cols + col + 1)
    plt.imshow(img)
    plt.title(f"{filename}\nLow-res", fontsize=9)
    plt.axis("off")

    # --- Plot mask ---
    plt.subplot(rows, cols, (base_row + 1) * cols + col + 1)
    plt.imshow(mask, cmap="gray")
    plt.title("Tissue mask", fontsize=9)
    plt.axis("off")

    # --- Plot masked image ---
    plt.subplot(rows, cols, (base_row + 2) * cols + col + 1)
    plt.imshow(img_masked)
    plt.title("Masked image", fontsize=9)
    plt.axis("off")

plt.tight_layout()
plt.savefig("preprocessing/pannel.png", dpi=250, bbox_inches="tight")
