import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constantes par défaut
DEFAULT_IMG_WIDTH = 256
DEFAULT_IMG_HEIGHT = 256
DEFAULT_LRG = 2e-4
DEFAULT_LRD = 2e-4
DEFAULT_BASE_HES_PATH = 'dataset_v2/HES'
DEFAULT_BASE_IHC_PATH = 'dataset_v2/CD30'
DEFAULT_CHECKPOINT_DIR = 'best_models'
DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 4
DEFAULT_LAMBDA = 10
DEFAULT_MODEL_SCALE = 0.75