import torch
import pandas as pd
import numpy as np
import os
import sys
import argparse
from PIL import Image
from tqdm import tqdm
from src import config
from torchvision import transforms
from src.models import Generator
from src.data_loader import create_dataloaders, get_image_paths, set_seed, RANDOM_SEED
import lpips
import random
import re
import csv


test_hes_path = "/work/imvia/in156281/cDDPMv2/dataset/test/HES"
test_ihc_path = "/work/imvia/in156281/cDDPMv2/dataset/test/CD30"
img_width = config.IMG_WIDTH
img_height = config.IMG_HEIGHT
batch_size = 32
checkpoint_path = "best_models/batch-32-scale-1-lambda-10-width-256-lrg-0.0001-lrd-0.0001.pth"
out_dir = "/work/imvia/in156281/cDDPMv2/dataset/test/virtual_cd30_GAN512"
img_range = "[-1, 1]"
num_workers = 0


device = config.device

# On ne travaille que sur le dossier test
test_hes, test_ihc = get_image_paths(test_hes_path, test_ihc_path)
print(f"Fichiers HES trouvés : {len(test_hes)}")
print(f"Fichiers CD30 trouvés : {len(test_ihc)}")
if len(test_hes) > 0:
    print("Exemple HES:", test_hes[0])
if len(test_ihc) > 0:
    print("Exemple CD30:", test_ihc[0])
n_test = len(test_hes)

# Dataset simple pour test
from torch.utils.data import Dataset, DataLoader
tfm = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
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
        return tfm(hes), tfm(ihc), self.ihc_list[idx]

test_ds = TestPairedDataset(test_hes, test_ihc)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# --- Load model ---
# Option: infer model_scale from ckpt filename if you want
match = re.search(r"scale-([0-9.]+)", os.path.basename(checkpoint_path))
model_scale = float(match.group(1)) if match else config.MODEL_SCALE

gen = Generator(model_scale=model_scale).to(device)
state = torch.load(checkpoint_path, map_location="cpu")

if isinstance(state, dict) and "generator" in state:
    state = state["generator"]

gen.load_state_dict(state)
gen.eval()

# --- Output dirs ---
ckpt_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
save_dir = out_dir
os.makedirs(save_dir, exist_ok=True)

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
            slide_id = os.path.basename(os.path.dirname(real_path))
            slide_out_dir = os.path.join(save_dir, slide_id)
            os.makedirs(slide_out_dir, exist_ok=True)
            out_name = f"{real_base}_VIRTUAL.png"
            out_path = os.path.join(slide_out_dir, out_name)
            img = (y_hat[i].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            Image.fromarray(img).save(out_path)
            rows.append([real_path, out_path])
print(f"Done. Saved {len(rows)} virtual images.")