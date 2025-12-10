import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_HES_PATH = 'dataset_v2/HES'
BASE_IHC_PATH = 'dataset_v2/CD30'
BATCH_SIZE = 16
IMG_WIDTH = 1024
IMG_HEIGHT = 1024
EPOCHS = 100
LAMBDA = 10
MODEL_SCALE = 0.75
CHECKPOINT_DIR = 'best_models'
LRG = 2e-4
LRD = 2e-4