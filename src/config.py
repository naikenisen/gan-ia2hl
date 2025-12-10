import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_HES_PATH = '/silver/ube/extraction_v1/HES'
BASE_IHC_PATH = '/silver/ube/extraction_v1/CD30'
BUFFER_SIZE = 400
BATCH_SIZE = 4
IMG_WIDTH = 1024
IMG_HEIGHT = 1024
EPOCHS = 100
LAMBDA = 10
MODEL_SCALE = 0.75
CHECKPOINT_DIR = 'best_models'
LRG = 2e-4
LRD = 2e-4