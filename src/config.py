import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_HandE_PATH = 'ia2hl_data/train/HES'
BASE_IHC_PATH = 'ia2hl_data/train/CD30'
BUFFER_SIZE = 400
BATCH_SIZE = 4
IMG_WIDTH = 1024
IMG_HEIGHT = 1024
EPOCHS = 100
LAMBDA = 100
MODEL_SCALE = 1
CHECKPOINT_DIR = 'best_models'
LRG = 2e-4
LRD = 2e-4