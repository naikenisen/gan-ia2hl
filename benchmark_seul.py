# benchmark.py (version: utilise CD30 virtuel déjà généré)
# pip install torchmetrics lpips tqdm pytorch-fid torchvision pillow

import os
import re
import torch
from PIL import Image
from tqdm import tqdm
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.fid import FrechetInceptionDistance
from src.data_loader import create_dataloaders, get_image_paths, set_seed, RANDOM_SEED
import lpips


# Main benchmarking function
def benchmark(ckpt, data_root, batch_size, num_workers, img_size, lpips_net,
              out_json, seed, img_range, virtual_dir):
    # Set random seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- IMPORTANT ---
    # On ne charge PLUS le modèle ici. On utilise des PNG virtuels déjà générés.
    # -----------------

    # Load dataset and DataLoader (identique à ton code)
    train_loader, valid_loader, test_loader = create_dataloaders(
        "dataset_gan_ia2hl/HES",
        "dataset_gan_ia2hl/CD30",
        img_size,
        img_size,
        batch_size
    )
    dataloader = test_loader  # shuffle=False dans create_dataloaders

    # Rebuild EXACT test file list to recover real filenames in the same order
    # (même logique que create_dataloaders)
    set_seed(RANDOM_SEED)
    hes_files, ihc_files = get_image_paths("dataset_gan_ia2hl/HES", "dataset_gan_ia2hl/CD30")

    import random
    combined = list(zip(hes_files, ihc_files))
    random.shuffle(combined)
    hes_files, ihc_files = zip(*combined)
    hes_files, ihc_files = list(hes_files), list(ihc_files)

    total_size = len(hes_files)
    train_size = int(0.7 * total_size)
    valid_size = int(0.15 * total_size)

    test_ihc_files = ihc_files[train_size + valid_size:]  # liste des CD30 réels du test set
    assert len(test_ihc_files) == len(dataloader.dataset), (
        f"Mismatch: test_ihc_files={len(test_ihc_files)} vs dataloader.dataset={len(dataloader.dataset)}"
    )

    # Transform pour lire les PNG virtuels en tensor [0,1]
    from torchvision import transforms
    to_tensor = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # [0,1]
    ])

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
        global_idx = 0  # index image dans le test set (ordre fixe)
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)

            # y = CD30 réel (normalisé [-1,1] par le dataset)
            # On le remet en [0,1] pour métriques
            if img_range == '[-1, 1]':
                y = (y + 1) / 2
            elif img_range == '[0, 1]':
                pass
            else:
                raise ValueError('Invalid range specified.')

            y = y.clamp(0, 1)

            # Construire y_hat depuis les PNG virtuels correspondants
            y_hat_list = []
            bs = y.shape[0]
            for i in range(bs):
                real_path = test_ihc_files[global_idx + i]
                real_base = os.path.splitext(os.path.basename(real_path))[0]
                virt_name = f"{real_base}_VIRTUAL.png"
                virt_path = os.path.join(virtual_dir, virt_name)

                if not os.path.exists(virt_path):
                    raise FileNotFoundError(f"Virtual image not found: {virt_path}")

                virt_img = Image.open(virt_path).convert("RGB")
                virt_tensor = to_tensor(virt_img)  # [0,1], CPU
                y_hat_list.append(virt_tensor)

            global_idx += bs

            y_hat = torch.stack(y_hat_list, dim=0).to(device)  # [B,3,H,W] en [0,1]
            y_hat = y_hat.clamp(0, 1)

            # Ensure 3 channels for metrics (au cas où y serait 1ch)
            y_3ch = y.repeat(1, 3, 1, 1) if y.shape[1] == 1 else y
            y_hat_3ch = y_hat.repeat(1, 3, 1, 1) if y_hat.shape[1] == 1 else y_hat

            # Update PSNR and SSIM metrics
            psnr_metric.update(y_hat_3ch, y_3ch)
            ssim_metric.update(y_hat_3ch, y_3ch)

            # Update FID metric
            fid_metric.update(y_3ch, real=True)
            fid_metric.update(y_hat_3ch, real=False)

            # Calculate LPIPS for this batch
            y_lpips = 2 * y_3ch - 1
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
        results = {'psnr': psnr_score, 'ssim': ssim_score, 'lpips': lpips_score, 'fid': fid_score}
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

    # Dossier où sont les PNG virtuels générés (celui affiché dans ton terminal)
    VIRTUAL_DIR = "inference/batch-32-scale-1.0-lambda-10.0-width-256-lrg-0.0001-lrd-0.0001/virtual_cd30_test"

    benchmark(CKPT, DATA_ROOT, BATCH_SIZE, NUM_WORKERS, IMG_SIZE, LPIPS_NET, OUT_JSON, SEED, IMG_RANGE, VIRTUAL_DIR)
