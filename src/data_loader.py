import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np

# Random seed pour la reproductibilité
RANDOM_SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
                
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

# Fonction pour créer les dataloaders
def create_dataloaders(base_hes_path, base_ihc_path, img_height, img_width, batch_size):
    # Initialiser le seed
    set_seed(RANDOM_SEED)
    
    hes_files, ihc_files = get_image_paths(base_hes_path, base_ihc_path)
    
    # Mélanger les données avec le seed
    combined = list(zip(hes_files, ihc_files))
    random.shuffle(combined)
    hes_files, ihc_files = zip(*combined)
    hes_files, ihc_files = list(hes_files), list(ihc_files)
    
    # Split train/validation/test (70%/15%/15%)
    total_size = len(hes_files)
    train_size = int(0.7 * total_size)
    valid_size = int(0.15 * total_size)
    
    train_hes = hes_files[:train_size]
    train_ihc = ihc_files[:train_size]
    
    valid_hes = hes_files[train_size:train_size + valid_size]
    valid_ihc = ihc_files[train_size:train_size + valid_size]
    
    test_hes = hes_files[train_size + valid_size:]
    test_ihc = ihc_files[train_size + valid_size:]
    
    # Créer les datasets
    train_dataset = IHCDataset(train_hes, train_ihc, img_height, img_width)
    valid_dataset = IHCDataset(valid_hes, valid_ihc, img_height, img_width)
    test_dataset = IHCDataset(test_hes, test_ihc, img_height, img_width)
    
    # Créer les dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, generator=torch.Generator().manual_seed(RANDOM_SEED))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, valid_loader, test_loader