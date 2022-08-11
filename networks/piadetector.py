import torch.nn as nn


class PiaDetector(nn.Module):

    def __init__(self, nb_features=1000):
        super().__init__()
        self.fc1 = nn.Linear(nb_features, 512)
        self.fc2 = nn.Linear(512, 100)
        self.fc3 = nn.Linear(100, 1)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.sig(out)
        out = self.fc2(out)
        out = self.sig(out)
        out = self.fc3(out)
        return out
