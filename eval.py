import os
import tempfile

# Configure cache directories BEFORE importing torch and matplotlib
# This prevents permission errors when downloading pretrained models
temp_dir = tempfile.gettempdir()
os.environ['TORCH_HOME'] = os.path.join(temp_dir, 'torch_cache')
os.environ['MPLCONFIGDIR'] = os.path.join(temp_dir, 'matplotlib_cache')
os.makedirs(os.environ['TORCH_HOME'], exist_ok=True)
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import sys
import lpips
import scipy.linalg
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from src.config import device, MODEL_SCALE, CHECKPOINT_DIR
from src.models import Generator
from src import data_loader

os.makedirs('results', exist_ok=True)
generator = Generator(MODEL_SCALE).to(device)
checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint_old.pth")
checkpoint = torch.load(checkpoint_path, map_location=device)
generator.load_state_dict(checkpoint['generator'])

# Initialize LPIPS model
lpips_model = lpips.LPIPS(net='alex').to(device)

def calculate_jsd(img1, img2, bins=256):
    """Calculate Jensen-Shannon Divergence between two images"""
    # Flatten images and compute histograms
    hist1, _ = np.histogram(img1.flatten(), bins=bins, range=(0, 255), density=True)
    hist2, _ = np.histogram(img2.flatten(), bins=bins, range=(0, 255), density=True)
    
    # Add small epsilon to avoid log(0)
    hist1 = hist1 + 1e-10
    hist2 = hist2 + 1e-10
    
    # Normalize to make them probability distributions
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    # Calculate JSD
    jsd = jensenshannon(hist1, hist2)
    return jsd

def calculate_fid_from_images(real_images, generated_images, device):
    """Calculate FID score using InceptionV3 features"""
    from torchvision.models import inception_v3
    from torch.nn import functional as F
    
    # Load InceptionV3 model in feature extraction mode
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.fc = torch.nn.Identity()  # Remove final classification layer
    inception_model.eval()
    
    def get_features(images):
        features = []
        with torch.no_grad():
            for img in images:
                # Resize to 299x299 for InceptionV3
                img_resized = F.interpolate(
                    img.unsqueeze(0), size=(299, 299), mode='bilinear', align_corners=False
                )
                # InceptionV3 expects images normalized to [-1, 1] which we already have
                feat = inception_model(img_resized)
                # feat is now a 1D tensor of features from the last pooling layer
                features.append(feat.squeeze().cpu().numpy())
        return np.array(features)
    
    # Get features
    real_features = get_features(real_images)
    gen_features = get_features(generated_images)
    
    # Calculate mean and covariance
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(gen_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(gen_features, rowvar=False)
    
    # Calculate FID
    diff = mu_real - mu_gen
    covmean = scipy.linalg.sqrtm(sigma_real.dot(sigma_gen))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid

def evaluate_model(model, train_loader, device, num_samples=20):
    print(f"Evaluating on {num_samples} samples")
    
    metrics = {
        'psnr': [],
        'ssim': [],
        'mae': [],
        'lpips': [],
        'jsd': []
    }
    
    # Store images for FID calculation
    real_images_for_fid = []
    generated_images_for_fid = []
    
    model.eval()
    lpips_model.eval()
    
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
            
            # Calculate LPIPS (expects normalized tensors in [-1, 1] range)
            # Take mean in case of batch, then convert to scalar
            lpips_value = lpips_model(ihc_img, generated_img).mean().item()
            
            # Calculate JSD
            jsd_value = calculate_jsd(ihc_img_denorm, generated_img_denorm)
            
            print(f"PSNR:  {psnr_value:.2f} dB")
            print(f"SSIM:  {ssim_value:.4f}")
            print(f"MAE:   {mae:.4f}")
            print(f"LPIPS: {lpips_value:.4f}")
            print(f"JSD:   {jsd_value:.4f}")
            
            metrics['psnr'].append(psnr_value)
            metrics['ssim'].append(ssim_value)
            metrics['mae'].append(mae)
            metrics['lpips'].append(lpips_value)
            metrics['jsd'].append(jsd_value)
            
            # Store images for FID calculation
            real_images_for_fid.append(ihc_img[0])
            generated_images_for_fid.append(generated_img[0])

    
    print(f"Evaluation complete: {len(metrics['psnr'])} samples processed")
    
    # Calculate FID score
    if len(real_images_for_fid) > 1:
        print("Calculating FID score...")
        fid_score = calculate_fid_from_images(real_images_for_fid, generated_images_for_fid, device)
        print(f"FID Score: {fid_score:.4f}")
        metrics['fid'] = [fid_score] * len(metrics['psnr'])  # Add FID as a constant for all samples
    
    return pd.DataFrame(metrics)


# Run evaluation
if __name__ == "__main__":
    print("STARTING EVALUATION")
    evaluation_df = evaluate_model(generator, data_loader_regions.train_loader, device, num_samples=20)
    print(evaluation_df.describe())
    metrics_path = 'results/evaluation_metrics.csv'
    evaluation_df.to_csv(metrics_path, index=False)
    print("EVALUATION COMPLETE")