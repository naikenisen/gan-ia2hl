import os
import re
import csv
import argparse
import random
import numpy as np
from PIL import Image

import torch
from tqdm import tqdm

from src import config
from src.models import Generator
from src.data_loader import get_image_paths, set_seed, RANDOM_SEED


def infer_testset_only(
    checkpoint_path: str,
    base_hes_path: str,
    base_ihc_path: str,
    img_height: int,
    img_width: int,
    batch_size: int,
    out_dir: str,
    img_range: str,
    num_workers: int = 0,
):
    device = config.device

    # --- Rebuild EXACT test split like create_dataloaders() ---
    set_seed(RANDOM_SEED)
    hes_files, ihc_files = get_image_paths(base_hes_path, base_ihc_path)

    combined = list(zip(hes_files, ihc_files))
    random.shuffle(combined)  # deterministic because seed set above
    hes_files, ihc_files = zip(*combined)
    hes_files, ihc_files = list(hes_files), list(ihc_files)

    total_size = len(hes_files)
    train_size = int(0.7 * total_size)
    valid_size = int(0.15 * total_size)
    test_hes = hes_files[train_size + valid_size:]
    test_ihc = ihc_files[train_size + valid_size:]

    assert len(test_hes) == len(test_ihc)
    n_test = len(test_hes)

    # --- Dataset that returns (x, y, ihc_path) so we can name outputs ---
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms

    tfm = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # -> [-1,1]
    ])

    class TestPairedDataset(Dataset):
        def __init__(self, hes_list, ihc_list):
            self.hes_list = hes_list
            self.ihc_list = ihc_list

        def __len__(self):
            return len(self.hes_list)

        def __getitem__(self, idx):
            hes = Image.open(self.hes_list[idx]).convert("RGB")
            ihc = Image.open(self.ihc_list[idx]).convert("RGB")
            return tfm(hes), tfm(ihc), self.ihc_list[idx]  # keep real path for naming

    test_ds = TestPairedDataset(test_hes, test_ihc)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # --- Load model ---
    # Option: infer model_scale from ckpt filename if you want
    match = re.search(r"scale-([0-9.]+)", os.path.basename(checkpoint_path))
    model_scale = float(match.group(1)) if match else config.DEFAULT_MODEL_SCALE

    gen = Generator(model_scale=model_scale).to(device)
    state = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(state, dict) and "generator" in state:
        state = state["generator"]

    gen.load_state_dict(state)
    gen.eval()

    # --- Output dirs ---
    ckpt_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    save_dir = os.path.join(out_dir, ckpt_name, "virtual_cd30_test")
    os.makedirs(save_dir, exist_ok=True)

    mapping_path = os.path.join(out_dir, ckpt_name, "mapping_test_virtual.csv")

    # --- Inference ---
    print(f"Inference on test set: {n_test} images -> {save_dir}")

    rows = []
    with torch.no_grad():
        for hes_imgs, ihc_imgs, ihc_paths in tqdm(test_loader, desc="Infer", unit="batch"):
            hes_imgs = hes_imgs.to(device, non_blocking=True)
            y_hat = gen(hes_imgs)

            # Convert to [0,1] for saving
            if img_range == "[-1, 1]":
                y_hat = (y_hat + 1) / 2
            elif img_range == "[0, 1]":
                pass
            else:
                raise ValueError("img_range must be '[-1, 1]' or '[0, 1]'")

            y_hat = y_hat.clamp(0, 1)

            # Save each generated image with name derived from real IHC filename
            for i in range(y_hat.size(0)):
                real_path = ihc_paths[i]
                real_base = os.path.splitext(os.path.basename(real_path))[0]

                # Example output name: <realname>_VIRTUAL.png
                out_name = f"{real_base}_VIRTUAL.png"
                out_path = os.path.join(save_dir, out_name)

                img = (y_hat[i].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                Image.fromarray(img).save(out_path)

                rows.append([real_path, out_path])

    # Save mapping csv (optional but very useful)
    os.makedirs(os.path.join(out_dir, ckpt_name), exist_ok=True)
    with open(mapping_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cd30_real_path", "cd30_virtual_path"])
        writer.writerows(rows)

    print(f"Done. Saved {len(rows)} virtual images.")
    print(f"Mapping saved to: {mapping_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_width", type=int, default=config.DEFAULT_IMG_WIDTH)
    parser.add_argument("--img_height", type=int, default=config.DEFAULT_IMG_HEIGHT)
    parser.add_argument("--base_hes_path", type=str, default=config.DEFAULT_BASE_HES_PATH)
    parser.add_argument("--base_ihc_path", type=str, default=config.DEFAULT_BASE_IHC_PATH)
    parser.add_argument("--batch_size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="inference")
    parser.add_argument("--img_range", type=str, default="[-1, 1]", choices=["[-1, 1]", "[0, 1]"])
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    infer_testset_only(
        checkpoint_path=args.checkpoint_path,
        base_hes_path=args.base_hes_path,
        base_ihc_path=args.base_ihc_path,
        img_height=args.img_height,
        img_width=args.img_width,
        batch_size=args.batch_size,
        out_dir=args.out_dir,
        img_range=args.img_range,
        num_workers=args.num_workers,
    )

