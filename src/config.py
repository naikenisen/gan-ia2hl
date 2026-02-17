import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMG_WIDTH = 256
IMG_HEIGHT = 256
LRG = 2e-4
LRD = 2e-4
BASE_HES_PATH = 'dataset_v2/HES'
BASE_IHC_PATH = 'dataset_v2/CD30'
CHECKPOINT_DIR = 'best_models'
BATCH_SIZE = 4
EPOCHS = 100
LAMBDA = 10
MODEL_SCALE = 0.75