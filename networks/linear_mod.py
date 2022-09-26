import torch
import torch.nn as nn


class DeepSigns(nn.Module):
    def __init__(self, gmm_mu):
        super().__init__()
        self.var_param = nn.Parameter(gmm_mu,
                                      requires_grad=True)

    def forward(self, matrix_a):
        matrix_g = torch.nn.Sigmoid()(self.var_param @ matrix_a)
        return matrix_g

    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f'var_param = {self.var_param.item()}'


class LinearMod(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config["n_features"], 1024, bias=False)
        self.fc2 = nn.Linear(1024, config["watermark_size"], bias=False)
        # self.fc3 = nn.Linear(config["watermark_size"], config["watermark_size"], bias=False)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.th = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.sig(out)
        out = self.fc2(out)
        out = self.sig(out)
        # out = self.fc3(out)
        # out = self.sig(out)
        return out
