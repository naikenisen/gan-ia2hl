import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import sys
from PIL import Image
from src.config import device, MODEL_SCALE, CHECKPOINT_DIR, IMG_HEIGHT, IMG_WIDTH
from torchvision import transforms
from src.models import Generator

hande_path = 'inference/hes.jpg'

os.makedirs('inference', exist_ok=True)
generator = Generator(MODEL_SCALE).to(device)
checkpoint_path = "checkpoints/checkpoint_1024.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
generator.load_state_dict(checkpoint['generator'])

transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

hande_img = transform(Image.open(hande_path).convert('RGB'))
# Add batch dimension: [C, H, W] -> [1, C, H, W]
hande_img = hande_img.unsqueeze(0)

generator.eval()

with torch.no_grad(): # disable gradient tracking for evaluation
        hande_img = hande_img.to(device)
        generated_img = generator(hande_img)
        
        # Move to CPU and convert to numpy
        hande_img_np = hande_img[0].cpu().numpy()
        generated_img_np = generated_img[0].cpu().numpy()
        
        # Denormalize images to be in the range [0, 255] for metric calculation
        hande_img_denorm = ((hande_img_np * 0.5 + 0.5) * 255.0).astype(np.uint8)
        generated_img_denorm = ((generated_img_np * 0.5 + 0.5) * 255.0).astype(np.uint8)
        
        # Convert from CHW to HWC for metrics
        generated_img_hwc = np.transpose(generated_img_denorm, (1, 2, 0))
        
        # Visualize the results
        plt.imsave('inference/inference.png', generated_img_hwc, format='png', cmap=None)
