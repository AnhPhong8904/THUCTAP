import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        dim = 32
        self.backbone = nn.Sequential(
            nn.Conv2d(3, dim, 3, stride=1, padding=1),  # 224x224
            nn.Conv2d(dim, dim, 3, stride=1, padding=1),  # 224x224
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112x112
            nn.Conv2d(dim, dim*2, 3, stride=1, padding=1),
            nn.Conv2d(dim*2, dim*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(dim*2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56x56
            nn.Conv2d(dim*2, dim*4, 3, stride=1, padding=1),
            nn.Conv2d(dim*4, dim*4, 3, stride=1, padding=1),
            nn.BatchNorm2d(dim*4),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28
            nn.Conv2d(dim*4, dim*8, 3, stride=1, padding=1),
            nn.Conv2d(dim*8, dim*8, 3, stride=1, padding=1),
            nn.BatchNorm2d(dim*8),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28
            nn.Conv2d(dim*8, dim*16, 3, stride=1, padding=1),
            nn.Conv2d(dim*16, dim*16, 3, stride=1, padding=1),
            nn.BatchNorm2d(dim*16),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Conv2d(dim*16, 5, 3, stride=1, padding=1), # 5 = 4 bbox + 1 obj confidence
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2)  # [B, 5, H, W]
        x = x.reshape(B, H*W, C)
        return x


if __name__ == "__main__":
    import torch, thop
    model = SimpleCNN()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)  # [1, 5, 14, 14]
    flops, params = thop.profile(model, inputs=(x,))
    print(f"FLOPS: {flops/1e9:.2f} GFLOPS, Params: {params/1e6:.2f}M")