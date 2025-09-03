import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        dim = 32
        self.features = nn.Sequential(
            nn.Conv2d(3, dim, 3, stride=1, padding=1),  # 224x224
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112x112
            nn.Conv2d(dim, dim*2, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56x56
            nn.Conv2d(dim*2, dim*4, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28
            nn.Conv2d(dim*4, dim*8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim*8 * 14 * 14, 512),
            nn.ReLU(),
            nn.Linear(512, 4),  # cx, cy, w, h
            nn.Sigmoid()  # ensure outputs in 0~1
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x
