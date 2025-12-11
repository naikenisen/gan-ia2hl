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
from src.data_loader import create_dataloaders

"""
Exemple d'utilisation:
python infer.py \
    --checkpoint_path best_models/best_model.pth \
    --img_width 256 \
    --img_height 256 \
    --batch_size 8 \
    --model_scale 0.75 \
    --base_hes_path dataset_v2/HES \
    --base_ihc_path dataset_v2/CD30
"""

parser = argparse.ArgumentParser()
parser.add_argument('--img_width', type=int, default=config.DEFAULT_IMG_WIDTH)
parser.add_argument('--img_height', type=int, default=config.DEFAULT_IMG_HEIGHT)
parser.add_argument('--model_scale', type=float, default=config.DEFAULT_MODEL_SCALE)
parser.add_argument('--base_hes_path', type=str, default=config.DEFAULT_BASE_HES_PATH)
parser.add_argument('--base_ihc_path', type=str, default=config.DEFAULT_BASE_IHC_PATH)
parser.add_argument('--batch_size', type=int, default=config.DEFAULT_BATCH_SIZE)
parser.add_argument('--checkpoint_path', type=str, default='best_models/best_model.pth')
args = parser.parse_args()

device = config.device

train_loader, valid_loader, test_loader = create_dataloaders(
    args.base_hes_path, 
    args.base_ihc_path,
    args.img_height,
    args.img_width,
    args.batch_size
)

os.makedirs('inference', exist_ok=True)
generator = Generator(args.model_scale).to(device)
checkpoint = torch.load(args.checkpoint_path, map_location=device)
generator.load_state_dict(checkpoint['generator'])

generator.eval()

print(f"Démarrage de l'inférence sur {len(test_loader)} batches du test set...")

batch_idx = 0
with torch.no_grad():
    for hes_imgs, ihc_imgs in test_loader:
        hes_imgs = hes_imgs.to(device)
        generated_imgs = generator(hes_imgs)
        
        # Sauvegarder chaque image du batch
        for i in range(generated_imgs.size(0)):
            # Image générée
            generated_img_np = generated_imgs[i].cpu().numpy()
            generated_img_denorm = ((generated_img_np * 0.5 + 0.5) * 255.0).astype(np.uint8)
            generated_img_hwc = np.transpose(generated_img_denorm, (1, 2, 0))
            
            # Image HES originale
            hes_img_np = hes_imgs[i].cpu().numpy()
            hes_img_denorm = ((hes_img_np * 0.5 + 0.5) * 255.0).astype(np.uint8)
            hes_img_hwc = np.transpose(hes_img_denorm, (1, 2, 0))
            
            # Image IHC cible (ground truth)
            ihc_img_np = ihc_imgs[i].cpu().numpy()
            ihc_img_denorm = ((ihc_img_np * 0.5 + 0.5) * 255.0).astype(np.uint8)
            ihc_img_hwc = np.transpose(ihc_img_denorm, (1, 2, 0))
            
            # Créer une figure avec les 3 images côte à côte
            img_id = batch_idx * args.batch_size + i
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(hes_img_hwc)
            axes[0].set_title('HES Original', fontsize=12)
            axes[0].axis('off')
            
            axes[1].imshow(generated_img_hwc)
            axes[1].set_title('IHC Généré', fontsize=12)
            axes[1].axis('off')
            
            axes[2].imshow(ihc_img_hwc)
            axes[2].set_title('IHC Target', fontsize=12)
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'inference/test_{img_id}_comparison.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        batch_idx += 1
        if batch_idx % 10 == 0:
            print(f"Traité {batch_idx}/{len(test_loader)} batches...")

print(f"Inférence terminée! {batch_idx * args.batch_size} images sauvegardées dans le dossier 'inference/'.")