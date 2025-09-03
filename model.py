import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        dim = 32
        self.features = nn.Sequential(
            nn.Conv2d(3, dim, 3, stride=2, padding=1),   # 112x112
            nn.ReLU(),
            nn.MaxPool2d(2), #56x56
            nn.Conv2d(dim, dim*2, 3, stride=2, padding=1), #28x28
            nn.ReLU(),
            nn.MaxPool2d(2), #14x14
            nn.Conv2d(dim*2, dim*4, 3, stride=2, padding=1), # 7x7
            nn.ReLU(),
            nn.MaxPool2d(2), #3x3
            nn.Conv2d(dim*4, dim*8, 3, stride=2, padding=1), #2x2
            nn.ReLU(),
            nn.Conv2d(dim*8, 4, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)   # (B,4,2,2)
        B,C,H,W = x.shape
        x = x.permute(0,2,3,1) # (B,H,W,C)
        x = x.view(B, H*W, C)  # (B,4,4)
        return x