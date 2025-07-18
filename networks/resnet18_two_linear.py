import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18TwoLinear(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.base = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Replace the original fc layer with two linear layers
        self.base.fc = nn.Sequential(
            nn.Linear(512, 256),  # First Linear
            nn.ReLU(),
            nn.Linear(256, num_classes)  # Second Linear
        )

    def forward(self, x):
        return self.base(x)