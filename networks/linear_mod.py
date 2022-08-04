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
        # self.matrix_a = nn.Parameter(2*torch.rand(size=(512, config["watermark_size"]))-1,
        #                              requires_grad=True)
        # self.var_param = nn.Parameter(torch.randn(size=(1, config["n_features"])),
        #                               requires_grad=True)
        self.fc = nn.Linear(config["n_features"], config["watermark_size"], bias=False)
        self.fc1 = nn.Linear(config["watermark_size"], config["watermark_size"], bias=False)
        # self.fc2 = nn.Linear(config["watermark_size"], config["watermark_size"], bias=False)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        # matrix_g = nn.Sigmoid()(var_param @ self.matrix_a)
        out = self.fc(x)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.sig(out)
        # out = self.fc2(out)
        # out = self.sig(out)
        return out
