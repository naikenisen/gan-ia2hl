import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import sys
from src.config import device, MODEL_SCALE, CHECKPOINT_DIR
from src.models import Generator
from src import data_loader_regions

os.makedirs('results', exist_ok=True)
generator = Generator(MODEL_SCALE).to(device)
checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint_old.pth")
checkpoint = torch.load(checkpoint_path, map_location=device)
generator.load_state_dict(checkpoint['generator'])

def evaluate_model(model, train_loader, device, num_samples=20):
    print(f"Evaluating on {num_samples} samples")
    
    metrics = {
        'psnr': [],
        'ssim': [],
        'mae': []
    }
    
    model.eval()
    
    with torch.no_grad(): # disable gradient tracking for evaluation
        for idx, (hande_img, ihc_img) in enumerate(train_loader):
            if idx >= num_samples:
                break
            
            print(f"Sample {idx + 1}/{num_samples}")
            
            hande_img = hande_img.to(device)
            ihc_img = ihc_img.to(device)
            
            generated_img = model(hande_img) # Generate IHC image from HandE image
            print(f"Image generated")
            
            # Move to CPU and convert to numpy
            hande_img_np = hande_img[0].cpu().numpy()
            ihc_img_np = ihc_img[0].cpu().numpy()
            generated_img_np = generated_img[0].cpu().numpy()
            
            # Denormalize images to be in the range [0, 255] for metric calculation
            hande_img_denorm = ((hande_img_np * 0.5 + 0.5) * 255.0).astype(np.uint8)
            ihc_img_denorm = ((ihc_img_np * 0.5 + 0.5) * 255.0).astype(np.uint8)
            generated_img_denorm = ((generated_img_np * 0.5 + 0.5) * 255.0).astype(np.uint8)
            
            # Convert from CHW to HWC for metrics
            ihc_img_hwc = np.transpose(ihc_img_denorm, (1, 2, 0))
            generated_img_hwc = np.transpose(generated_img_denorm, (1, 2, 0))
            
            # Calculate metrics
            psnr_value = psnr(ihc_img_hwc, generated_img_hwc, data_range=255)
            ssim_value = ssim(ihc_img_hwc, generated_img_hwc, channel_axis=2, data_range=255)
            mae = np.mean(np.abs(ihc_img_np - generated_img_np))
            
            print(f"PSNR: {psnr_value:.2f} dB")
            print(f"SSIM: {ssim_value:.4f}")
            print(f"MAE:  {mae:.4f}")
            
            metrics['psnr'].append(psnr_value)
            metrics['ssim'].append(ssim_value)
            metrics['mae'].append(mae)

    
    print(f"Evaluation complete: {len(metrics['psnr'])} samples processed")
    return pd.DataFrame(metrics)


# Run evaluation
if __name__ == "__main__":
    print("STARTING EVALUATION")
    evaluation_df = evaluate_model(generator, data_loader_regions.train_loader, device, num_samples=20)
    print(evaluation_df.describe())
    metrics_path = 'results/evaluation_metrics.csv'
    evaluation_df.to_csv(metrics_path, index=False)
    print("EVALUATION COMPLETE")