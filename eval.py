import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import sys

print("=" * 60)
print("EVALUATION SCRIPT START")
print("=" * 60)

# Import configuration
print("\n[1/6] Importing configuration...")
try:
    from src.config import device, MODEL_SCALE, CHECKPOINT_DIR
    print(f"   [OK] Device: {device}")
    print(f"   [OK] Model scale: {MODEL_SCALE}")
    print(f"   [OK] Checkpoint dir: {CHECKPOINT_DIR}")
except Exception as e:
    print(f"   [ERROR] Failed to import config: {e}")
    sys.exit(1)

# Import models
print("\n[2/6] Importing models...")
try:
    from src.models import Generator
    print("   [OK] Generator imported")
except Exception as e:
    print(f"   [ERROR] Failed to import Generator: {e}")
    sys.exit(1)

# Import data loader
print("\n[3/6] Importing test_loader...")
try:
    from src.data_loader import test_loader
    print(f"   [OK] test_loader imported ({len(test_loader)} batches)")
except Exception as e:
    print(f"   [ERROR] Failed to import test_loader: {e}")
    sys.exit(1)

# Create results folder
print("\n[4/6] Creating results folder...")
os.makedirs('results', exist_ok=True)
print("   [OK] Folder 'results' created/verified")

# Create generator
print("\n[5/6] Creating generator...")
try:
    generator = Generator(MODEL_SCALE).to(device)
    print(f"   [OK] Generator created and placed on {device}")
except Exception as e:
    print(f"   [ERROR] Failed to create generator: {e}")
    sys.exit(1)

# Load weights from checkpoint
print("\n[6/6] Loading checkpoint...")
checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")
print(f"   [INFO] Path: {checkpoint_path}")

if not os.path.exists(checkpoint_path):
    print(f"   [ERROR] File does not exist!")
    print(f"   Check that training has created a checkpoint.")
    sys.exit(1)

try:
    print(f"   [INFO] Loading...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"   [OK] Checkpoint loaded")
    print(f"   [INFO] Checkpoint epoch: {checkpoint['epoch']}")
    
    generator.load_state_dict(checkpoint['generator'])
    print(f"   [OK] Generator weights loaded")
except Exception as e:
    print(f"   [ERROR] Failed to load checkpoint: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("INITIALIZATION COMPLETE - Starting evaluation")
print("=" * 60 + "\n")

def evaluate_model(model, test_loader, device, num_samples=20):
    print(f"\nEvaluating on {num_samples} samples...")
    
    metrics = {
        'psnr': [],
        'ssim': [],
        'mae': []
    }
    
    model.eval() # Set model to evaluation mode
    print("   [OK] Model in evaluation mode")
    
    with torch.no_grad(): # disable gradient tracking for evaluation
        for idx, (hande_img, ihc_img) in enumerate(test_loader):
            if idx >= num_samples:
                break
            
            print(f"\n   [INFO] Sample {idx + 1}/{num_samples}")
            
            hande_img = hande_img.to(device)
            ihc_img = ihc_img.to(device)
            print(f"      - Images loaded on {device}")
            print(f"      - Shape: {hande_img.shape}")
            
            generated_img = model(hande_img) # Generate IHC image from HandE image
            print(f"      - Image generated")
            
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
            try:
                psnr_value = psnr(ihc_img_hwc, generated_img_hwc, data_range=255)
                ssim_value = ssim(ihc_img_hwc, generated_img_hwc, channel_axis=2, data_range=255)
                mae = np.mean(np.abs(ihc_img_np - generated_img_np))
                
                print(f"      - PSNR: {psnr_value:.2f} dB")
                print(f"      - SSIM: {ssim_value:.4f}")
                print(f"      - MAE:  {mae:.4f}")
                
                metrics['psnr'].append(psnr_value)
                metrics['ssim'].append(ssim_value)
                metrics['mae'].append(mae)
            except Exception as e:
                print(f"      [ERROR] Failed to calculate metrics: {e}")
                continue
            
            # Visualize the results
            plt.figure(figsize=(18, 7))
            
            plt.subplot(1, 3, 1)
            plt.imshow(np.transpose(hande_img_denorm, (1, 2, 0)))
            plt.title("Input (HandE) Image")
            plt.axis("off")
            
            plt.subplot(1, 3, 2)
            plt.imshow(generated_img_hwc)
            plt.title("Generated (IHC) Image")
            plt.axis("off")
            
            plt.subplot(1, 3, 3)
            plt.imshow(ihc_img_hwc)
            plt.title("Ground Truth (IHC) Image")
            plt.axis("off")
            
            output_path = f'results/sample_{idx}.png'
            plt.savefig(output_path, bbox_inches='tight', dpi=1000)
            plt.close()
            print(f"      [OK] Image saved: {output_path}")
    
    print(f"\n[OK] Evaluation complete: {len(metrics['psnr'])} samples processed\n")
    return pd.DataFrame(metrics)


# Run evaluation
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("STARTING EVALUATION")
    print("=" * 60)
    
    # Run evaluation on 20 samples
    evaluation_df = evaluate_model(generator, test_loader, device, num_samples=20)
    
    # Display summary statistics
    print("\n" + "=" * 60)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 60)
    print(evaluation_df.describe())
    
    # Save metrics to CSV
    metrics_path = 'results/evaluation_metrics.csv'
    evaluation_df.to_csv(metrics_path, index=False)
    print(f"\n[OK] Metrics saved to: {metrics_path}")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)