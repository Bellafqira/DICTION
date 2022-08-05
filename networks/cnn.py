import torch
import torch.nn as nn


class CnnModel(nn.Module):
    """lr=0.001, opt=Adam, """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3))
        # pooling
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3))
        # pooling
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(in_features=64 * 5 * 5, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)
        self.relu = nn.ReLU()
        self.bt_n = nn.BatchNorm1d(1600)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.pool(self.conv2(out)))
        out = self.relu(self.conv3(out))
        out = self.relu(self.pool(self.conv4(out)))
        out = torch.flatten(out, 1)  # flatten all dimensions except batch
        out = self.bt_n(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

#
# you can use this formula [(W−K+2P)/S]+1.
# W is the input volume - in your case 128
# K is the Kernel size - in your case 5
# P is the padding - in your case 0 i believe
# S is the stride - which you have not provided.
