import torch


T_SIZE = 300
M_SIZE = 80
BATCH_SIZE = 32
NUM_RES_BLOCKS = 3
HIDDEN_DIM = 4
LR = 0.0002
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
