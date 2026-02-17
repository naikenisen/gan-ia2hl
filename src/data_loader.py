import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
import glob

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
    # Supporte .jpg, .jpeg, .png (insensible à la casse)
    exts = ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')
    hes_files = [f for f in os.listdir(hes_base) if f.endswith(exts)]
    hes_files.sort()  # pour l'ordre
    for img_name in hes_files:
        hes_path = os.path.join(hes_base, img_name)
        ihc_path = os.path.join(ihc_base, img_name)
        if os.path.exists(ihc_path):
            hes_paths.append(hes_path)
            ihc_paths.append(ihc_path)
    return hes_paths, ihc_paths

class IHCDataset(Dataset):
    def __init__(self, hes_paths, ihc_paths, img_height, img_width):
        self.hes_paths = hes_paths
        self.ihc_paths = ihc_paths
        # On définit la transformation ici
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

def get_sorted_files(folder_path):
    """
    Récupère tous les fichiers images d'un dossier et les trie
    pour assurer la correspondance HES <-> CD30.
    """
    # On cherche les extensions courantes
    extensions = ['*.jpg', '*.png', '*.jpeg', '*.tif']
    files = []
    for ext in extensions:
        # recursive=False car on suppose que les images sont directement dans le dossier
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    # Le tri est INDISPENSABLE pour que l'image HES corresponde à la bonne image CD30
    return sorted(files)

def create_dataloaders(dataset_root, img_height, img_width, batch_size):
    
    # 1. Définition des chemins
    # On utilise os.path.join pour être compatible Windows/Linux
    train_hes_dir = "dataset/train/HES"
    train_ihc_dir = "dataset/train/CD30"
    
    valid_hes_dir = "dataset/valid/HES"
    valid_ihc_dir = "dataset/valid/CD30"
    
    test_hes_dir = "dataset/test/HES"
    test_ihc_dir = "dataset/test/CD30"

    # 2. Récupération des fichiers
    train_hes = get_sorted_files(train_hes_dir)
    train_ihc = get_sorted_files(train_ihc_dir)
    
    valid_hes = get_sorted_files(valid_hes_dir)
    valid_ihc = get_sorted_files(valid_ihc_dir)
    
    test_hes = get_sorted_files(test_hes_dir)
    test_ihc = get_sorted_files(test_ihc_dir)

    # 3. Création des Datasets
    train_dataset = IHCDataset(train_hes, train_ihc, img_height, img_width)
    valid_dataset = IHCDataset(valid_hes, valid_ihc, img_height, img_width)
    test_dataset = IHCDataset(test_hes, test_ihc, img_height, img_width)
    
    # 4. Création des Dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Important pour le train
        num_workers=2,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    return train_loader, valid_loader, test_loader