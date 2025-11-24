import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_HandE_PATH = '/work/imvia/in156281/ihc4bc/IHC4BC_Compressed/Images/HandE'
BASE_IHC_PATH = '/work/imvia/in156281/ihc4bc/IHC4BC_Compressed/Images/IHC'
RECEPTOR = 'Ki67'
BUFFER_SIZE = 400
BATCH_SIZE = 4
IMG_WIDTH = 256
IMG_HEIGHT = 256
EPOCHS = 100
LAMBDA = 100
MODEL_SCALE = 0.75
CHECKPOINT_DIR = '/work/imvia/in156281/ihc4bc/checkpoints'