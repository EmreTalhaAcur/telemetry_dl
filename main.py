import torch
import torch.nn as nn
import torch.optim as optim

# Kendi yazdığımız modüllerden import ediyoruz
from src.config import DEVICE, LEARNING_RATE
from src.dataset import get_data_loaders
from src.model import TelemetryModel
from src.trainer import train_model

def main():
    # 1. Veriyi hazırla
    data_loader = get_data_loaders()
    
    # 2. Modeli ayağa kaldır ve cihaza (CPU/GPU) yükle
    model = TelemetryModel().to(DEVICE)
    
    # 3. Kayıp Fonksiyonu ve Optimizer'ı tanımla
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Eğitimi Başlat
    train_model(model, data_loader, criterion, optimizer, DEVICE)
    
    # 5. Modeli Kaydet (Senior Alışkanlığı)
    torch.save(model.state_dict(), "telemetry_model.pt")
    print("\nModel başarıyla 'telemetry_model.pt' olarak kaydedildi!")

if __name__ == "__main__":
    main()