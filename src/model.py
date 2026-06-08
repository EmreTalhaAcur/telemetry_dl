import torch.nn as nn

class TelemetryModel(nn.Module):
    def __init__(self):
        super(TelemetryModel, self).__init__()
        # Girişte 2 özellik var (RPM, Yük) -> Gizli katmanda 16 nörona bağlanıyor
        self.gizli_katman = nn.Linear(2, 16)
        self.aktivasyon = nn.ReLU()
        # 16 nörondan tek bir çıkış alıyoruz (Tahmin edilen Motor Sıcaklığı)
        self.cikis_katman = nn.Linear(16, 1)

    def forward(self, x):
        x = self.gizli_katman(x)
        x = self.aktivasyon(x)
        x = self.cikis_katman(x)
        return x