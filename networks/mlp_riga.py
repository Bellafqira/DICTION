import torch
import torch.nn as nn


class MLP_RIGA(nn.Module):
    """lr=0.001, opt=Adam, """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=(3, 3))

        self.fc1 = nn.Linear(in_features=13824, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)

        self.bt_n_1 = nn.BatchNorm2d(24)
        self.bt_n_2 = nn.BatchNorm2d(24)

        self.soft = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.bt_n_1(out)
        out = self.relu(self.conv2(out))
        out = self.bt_n_2(out)

        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        # here to watermark
        out = self.fc3(out)
        out = self.soft(out)
        return out

#
# you can use this formula [(W−K+2P)/S]+1.
# W is the input volume - in your case 128
# K is the Kernel size - in your case 5
# P is the padding - in your case 0 i believe
# S is the stride - which you have not provided.
