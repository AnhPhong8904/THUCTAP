import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, max_objs, dim=32):
        super(MyModel, self).__init__()
        self.max_objs = max_objs
        self.dim = dim
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
        
        self.features = nn.Sequential(
            nn.Conv2d(dim*16, dim*32, 3, stride=1, padding=1),
            nn.BatchNorm2d(dim*32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(dim*32, 512),
            nn.ReLU(),
            nn.Linear(512, (17 * 3 + 1) *  self.max_objs)
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.features(x)
        x = self.regressor(x)
        x = x.view(-1, self.max_objs, 17 * 3 + 1)  # Reshape to (batch_size, max_objs, 17 keypoints, 3 values)
        return x
    

if __name__ == "__main__":
    import torch, thop
    model = MyModel(max_objs=1)
    input = torch.randn(1, 3, 224, 224)
    flops, params = thop.profile(model, inputs=(input, ), verbose=False)
    print("FLOPs: %.2fG, Params: %.2fM" % (flops/1e9, params/1e6))
    y = model(input)
    print(y.shape)