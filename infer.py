import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
from PIL import Image
from src import config
from torchvision import transforms
from src.models import Generator

# Parser les arguments
parser = argparse.ArgumentParser(description='Inference with trained pix2pix GAN')
parser.add_argument('--img_width', type=int, default=config.DEFAULT_IMG_WIDTH, help='Image width')
parser.add_argument('--img_height', type=int, default=config.DEFAULT_IMG_HEIGHT, help='Image height')
parser.add_argument('--model_scale', type=float, default=config.DEFAULT_MODEL_SCALE, help='Model scale factor')
parser.add_argument('--checkpoint_path', type=str, default='best_models/best_model.pth', help='Path to checkpoint')
args = parser.parse_args()

device = config.device

hes_train_path = 'inference/original_hes_train.jpg'
hes_test_path = 'inference/original_hes_test.jpg'

os.makedirs('inference', exist_ok=True)
generator = Generator(args.model_scale).to(device)
checkpoint = torch.load(args.checkpoint_path, map_location=device)
generator.load_state_dict(checkpoint['generator'])

transform = transforms.Compose([
            transforms.Resize((args.img_height, args.img_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

hes_train_img = transform(Image.open(hes_train_path).convert('RGB'))
hes_train_img = hes_train_img.unsqueeze(0)

hes_test_img = transform(Image.open(hes_test_path).convert('RGB'))
hes_test_img = hes_test_img.unsqueeze(0)

generator.eval()

with torch.no_grad():
        # Train image
        hes_train_img = hes_train_img.to(device)
        generated_train_img = generator(hes_train_img)
        generated_train_img_np = generated_train_img[0].cpu().numpy()
        generated_train_img_denorm = ((generated_train_img_np * 0.5 + 0.5) * 255.0).astype(np.uint8)
        generated_train_img_hwc = np.transpose(generated_train_img_denorm, (1, 2, 0))
        plt.imsave('inference/inference_train.png', generated_train_img_hwc, format='png', cmap=None)
        
        # Test image
        hes_test_img = hes_test_img.to(device)
        generated_test_img = generator(hes_test_img)
        generated_test_img_np = generated_test_img[0].cpu().numpy()
        generated_test_img_denorm = ((generated_test_img_np * 0.5 + 0.5) * 255.0).astype(np.uint8)
        generated_test_img_hwc = np.transpose(generated_test_img_denorm, (1, 2, 0))
        plt.imsave('inference/inference_test.png', generated_test_img_hwc, format='png', cmap=None)

# afficher les images côte à côte avec quadrillage et coordonnées
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
# Charger les images
img1 = mpimg.imread('inference/original_cd30_train.jpg')
img2 = mpimg.imread('inference/inference_train.png')
img3 = mpimg.imread('inference/original_cd30_test.jpg')
img4 = mpimg.imread('inference/inference_test.png')
# Créer la figure avec 4 sous-plots côte à côte
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
# Espacer légèrement les images
plt.subplots_adjust(wspace=0.05, top=0.85)

# Titres pour chaque image
titles = ['IHC Original (Train)', 'IHC Synthétique (Train)', 
          'IHC Original (Test)', 'IHC Synthétique (Test)']
# Paramètres du quadrillage
grid_spacing = 10  # Nombre de lignes/colonnes dans le quadrillage
# Afficher les images et ajouter le quadrillage
for idx, ax in enumerate(axes):
    if idx == 0:
        ax.imshow(img1)
    elif idx == 1:
        ax.imshow(img2)
    elif idx == 2:
        ax.imshow(img3)
    else:
        ax.imshow(img4)
    
    # Ajouter le titre
    ax.set_title(titles[idx], fontsize=14, weight='bold', pad=20)
    
    # Obtenir les dimensions de l'image affichée
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Créer le quadrillage
    x_step = (xlim[1] - xlim[0]) / grid_spacing
    y_step = (ylim[0] - ylim[1]) / grid_spacing
    # Lignes verticales
    for i in range(grid_spacing + 1):
        x = xlim[0] + i * x_step
        ax.axvline(x=x, color='white', linewidth=0.8, alpha=0.8)
    # Lignes horizontales
    for i in range(grid_spacing + 1):
        y = ylim[1] + i * y_step
        ax.axhline(y=y, color='white', linewidth=0.8, alpha=0.8)
    # Ajouter les coordonnées sur les axes
    # Coordonnées X (en haut)
    x_positions = [xlim[0] + (i + 0.5) * x_step for i in range(grid_spacing)]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(range(grid_spacing), fontsize=10, color='black', weight='bold')
    ax.tick_params(axis='x', length=0, pad=5, color='black', labelcolor='black')
    ax.xaxis.tick_top()
    # Coordonnées Y (à gauche)
    y_positions = [ylim[1] + (i + 0.5) * y_step for i in range(grid_spacing)]
    ax.set_yticks(y_positions)
    ax.set_yticklabels(range(grid_spacing), fontsize=10, color='black', weight='bold')
    ax.tick_params(axis='y', length=0, pad=5, color='black', labelcolor='black')
    # S'assurer que les axes sont visibles
    ax.spines['top'].set_visible(True)
    ax.spines['left'].set_visible(True)
# Sauvegarder la figure
fig.savefig('inference/view_images_grid.png', dpi=300, bbox_inches='tight')