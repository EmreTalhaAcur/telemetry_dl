import torch

# Hiperparametreler
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.01
SEED = 42

# Donanım Seçimi
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")