import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMG_WIDTH = 256
IMG_HEIGHT = 256
LRG = 1e-4
LRD = 1e-4
BASE_HES_PATH = 'dataset/HES'
BASE_IHC_PATH = 'dataset/CD30'
CHECKPOINT_DIR = 'best_models'
BATCH_SIZE = 32
EPOCHS = 50
LAMBDA = 10
MODEL_SCALE = 1