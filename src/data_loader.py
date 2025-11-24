import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from src.config import BASE_HandE_PATH, BASE_IHC_PATH, RECEPTOR, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE
                
# fonction pour récupérer les chemins des images HandE et IHC
def get_image_paths(hande_base, ihc_base, receptor):
    hande_paths, ihc_paths = [], []
    receptor_hande_dir = os.path.join(hande_base, receptor)
    receptor_ihc_dir = os.path.join(ihc_base, receptor)
    for patient_dir in os.listdir(receptor_hande_dir):
        patient_hande_path = os.path.join(receptor_hande_dir, patient_dir)
        patient_ihc_path = os.path.join(receptor_ihc_dir, patient_dir)
        if not os.path.isdir(patient_hande_path): continue
        for subregion_dir in os.listdir(patient_hande_path):
            subregion_hande_path = os.path.join(patient_hande_path, subregion_dir)
            subregion_ihc_path = os.path.join(patient_ihc_path, subregion_dir)
            if not os.path.isdir(subregion_hande_path): continue
            for img_name in os.listdir(subregion_hande_path):
                if img_name.endswith('.jpg'):
                    hande_paths.append(os.path.join(subregion_hande_path, img_name))
                    ihc_paths.append(os.path.join(subregion_ihc_path, img_name))
    return hande_paths, ihc_paths

# fonction pour prétraiter les images et créer un Dataset personnalisé
class IHCDataset(Dataset):
    def __init__(self, hande_paths, ihc_paths, img_height, img_width):
        self.hande_paths = hande_paths
        self.ihc_paths = ihc_paths
        self.transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.hande_paths)
    
    def __getitem__(self, idx):
        hande_img = Image.open(self.hande_paths[idx]).convert('RGB')
        ihc_img = Image.open(self.ihc_paths[idx]).convert('RGB')
        hande_img = self.transform(hande_img)
        ihc_img = self.transform(ihc_img)
        return hande_img, ihc_img

hande_files, ihc_files = get_image_paths(BASE_HandE_PATH, BASE_IHC_PATH, RECEPTOR)
train_size = int(0.8 * len(hande_files))
train_hande, test_hande = hande_files[:train_size], hande_files[train_size:]
train_ihc, test_ihc = ihc_files[:train_size], ihc_files[train_size:]

train_dataset = IHCDataset(train_hande, train_ihc, IMG_HEIGHT, IMG_WIDTH)
test_dataset = IHCDataset(test_hande, test_ihc, IMG_HEIGHT, IMG_WIDTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)