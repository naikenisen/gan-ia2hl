import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_HandE_PATH = 'ia2hl_data/train/HES'
BASE_IHC_PATH = 'ia2hl_data/train/CD30'
BUFFER_SIZE = 400
BATCH_SIZE = 4
IMG_WIDTH = 512
IMG_HEIGHT = 512
EPOCHS = 100
LAMBDA = 100
MODEL_SCALE = 0.75
CHECKPOINT_DIR = 'checkpoints'