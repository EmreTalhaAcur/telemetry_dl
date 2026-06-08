import torch
from src.config import EPOCHS

def train_model(model, data_loader, criterion, optimizer, device):
    """Modeli eğitir ve her epoch sonundaki durumu raporlar."""
    print(f"\n--- Model Eğitimi Başlıyor ({device} üzerinde) ---")
    
    model.train() # Modeli eğitim moduna al
    for epoch in range(1, EPOCHS + 1):
        toplam_hata = 0.0
        
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Eğitim adımları
            optimizer.zero_grad()
            tahminler = model(batch_X)
            loss = criterion(tahminler, batch_y)
            loss.backward()
            optimizer.step()
            
            toplam_hata += loss.item()
            
        if epoch % 5 == 0 or epoch == 1:
            ortalama_hata = toplam_hata / len(data_loader)
            print(f"Epoch: {epoch:02d}/{EPOCHS} | Ortalama Hata: {ortalama_hata:.4f}")