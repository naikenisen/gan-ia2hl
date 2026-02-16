# benchmark.py

# Dependencies
# Install with: 
# pip install torchmetrics lpips tqdm
# pip install pytorch-fid

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from src.models import Generator
from src.data_loader import create_dataloaders
import lpips


# TODO: Import your model and dataset
# from myproject.models import MyModel
# from myproject.data import MyDataset

# Main benchmarking function
def benchmark(ckpt, data_root, batch_size, num_workers, img_size, lpips_net, out_json, seed, img_range):
    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    # Load model checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# (Optionnel mais cohérent) récupérer le scale depuis le nom du ckpt
    import os, re
    match = re.search(r'scale-([0-9.]+)', os.path.basename(ckpt))
    model_scale = float(match.group(1)) if match else 0.75

    model = Generator(model_scale=model_scale).to(device)

    state = torch.load(ckpt, map_location="cpu")

    # ton ckpt peut être un dict avec "generator"
    if isinstance(state, dict) and "generator" in state:
        state = state["generator"]

    model.load_state_dict(state)
    model.eval()

    # Load dataset and DataLoader
    train_loader, valid_loader, test_loader = create_dataloaders(
    "dataset_gan_ia2hl/HES",
    "dataset_gan_ia2hl/CD30",
    img_size,
    img_size,
    batch_size
    )

    dataloader = test_loader


    # Initialize metrics once
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    lpips_model = lpips.LPIPS(net=lpips_net).to(device)
    lpips_model.eval()

    lpips_sum = 0.0
    num_images = 0

    # Loop through DataLoader
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            y_hat = model(x)

            # Normalize/clip images based on range
            if img_range == '[-1, 1]':
                y_hat = (y_hat + 1) / 2  # Convert to [0, 1]
                y = (y + 1) / 2  # Convert to [0, 1]
            elif img_range == '[0, 1]':
                pass  # Already in [0, 1]
            else:
                raise ValueError('Invalid range specified.')

            # Clamp values to [0, 1]
            y_hat = y_hat.clamp(0, 1)
            y = y.clamp(0, 1)

            # Ensure 3 channels for metrics
            y_3ch = y.repeat(1, 3, 1, 1) if y.shape[1] == 1 else y
            y_hat_3ch = y_hat.repeat(1, 3, 1, 1) if y_hat.shape[1] == 1 else y_hat

            # Update PSNR and SSIM metrics
            psnr_metric.update(y_hat_3ch, y_3ch)
            ssim_metric.update(y_hat_3ch, y_3ch)

            # Update FID metric
            fid_metric.update(y_3ch, real=True)
            fid_metric.update(y_hat_3ch, real=False)

            # Calculate LPIPS for this batch
            y_lpips = 2 * y_3ch - 1  # Convert [0, 1] to [-1, 1]
            y_hat_lpips = 2 * y_hat_3ch - 1
            lpips_batch = lpips_model(y_hat_lpips, y_lpips).mean()
            lpips_sum += lpips_batch.item() * y.shape[0]
            num_images += y.shape[0]

    # Compute final metrics
    psnr_score = psnr_metric.compute().item()
    ssim_score = ssim_metric.compute().item()
    fid_score = fid_metric.compute().item()
    lpips_score = lpips_sum / num_images

    # Summary
    print(f'Number of images evaluated: {num_images}')
    print(f'Mean PSNR: {psnr_score:.2f}')
    print(f'Mean SSIM: {ssim_score:.2f}')
    print(f'Mean LPIPS: {lpips_score:.2f}')
    print(f'FID: {fid_score:.2f}')

    # Save results to JSON if specified
    if out_json:
        import json
        results = {
            'psnr': psnr_score,
            'ssim': ssim_score,
            'lpips': lpips_score,
            'fid': fid_score
        }
        with open(out_json, 'w') as f:
            json.dump(results, f)

if __name__ == '__main__':
    # Configuration
    CKPT = "best_models/batch-32-scale-1.0-lambda-10.0-width-256-lrg-0.0001-lrd-0.0001.pth"
    DATA_ROOT = "data/test"
    BATCH_SIZE = 4
    NUM_WORKERS = 0
    IMG_SIZE = 256
    LPIPS_NET = "alex"
    OUT_JSON = None
    SEED = None
    IMG_RANGE = "[-1, 1]"
    
    benchmark(CKPT, DATA_ROOT, BATCH_SIZE, NUM_WORKERS, IMG_SIZE, LPIPS_NET, OUT_JSON, SEED, IMG_RANGE)
