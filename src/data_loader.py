import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from src.config import BASE_HES_PATH, BASE_IHC_PATH, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE
                
# fonction pour récupérer les chemins des images hes et IHC
def get_image_paths(hes_base, ihc_base):
    hes_paths, ihc_paths = [], []
    for patient_dir in os.listdir(hes_base):
        patient_hes_path = os.path.join(hes_base, patient_dir)
        patient_ihc_path = os.path.join(ihc_base, patient_dir)
        if not os.path.isdir(patient_hes_path): continue
        # Parcourir directement les images dans le dossier du patient
        for img_name in os.listdir(patient_hes_path):
            if img_name.endswith('.jpg'):
                hes_paths.append(os.path.join(patient_hes_path, img_name))
                ihc_paths.append(os.path.join(patient_ihc_path, img_name))
    return hes_paths, ihc_paths

# fonction pour prétraiter les images et créer un Dataset personnalisé
class IHCDataset(Dataset):
    def __init__(self, hes_paths, ihc_paths, img_height, img_width):
        self.hes_paths = hes_paths
        self.ihc_paths = ihc_paths
        self.transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.hes_paths)
    
    def __getitem__(self, idx):
        hes_img = Image.open(self.hes_paths[idx]).convert('RGB')
        ihc_img = Image.open(self.ihc_paths[idx]).convert('RGB')
        hes_img = self.transform(hes_img)
        ihc_img = self.transform(ihc_img)
        return hes_img, ihc_img

hes_files, ihc_files = get_image_paths(BASE_HES_PATH, BASE_IHC_PATH)
train_size = int(0.8 * len(hes_files))
train_hes, valid_hes = hes_files[:train_size], hes_files[train_size:]
train_ihc, valid_ihc = ihc_files[:train_size], ihc_files[train_size:]

train_dataset = IHCDataset(train_hes, train_ihc, IMG_HEIGHT, IMG_WIDTH)
valid_dataset = IHCDataset(valid_hes, valid_ihc, IMG_HEIGHT, IMG_WIDTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)