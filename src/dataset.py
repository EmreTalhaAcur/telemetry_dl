import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.config import SEED, BATCH_SIZE

class TelemetryDataset(Dataset):
    def __init__(self):
        np.random.seed(SEED)
        # 1000 adet yapay sensör verisi [RPM, Yük]
        self.X_raw = np.random.rand(1000, 2).astype(np.float32)
        self.y_raw = (self.X_raw[:, 0] * 50 + self.X_raw[:, 1] * 30 + 40).reshape(-1, 1).astype(np.float32)

        self.X = torch.from_numpy(self.X_raw)
        self.y = torch.from_numpy(self.y_raw)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_data_loaders():
    """DataLoader'ı dış dünyaya servis eden yardımcı fonksiyon."""
    dataset = TelemetryDataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader