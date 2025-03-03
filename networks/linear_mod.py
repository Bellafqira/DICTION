import torch
import torch.nn as nn

from util.util import Random


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


class Uchida(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.var_param = nn.Parameter(w, requires_grad=True)

    def forward(self, matrix_a):
        matrix_g = torch.nn.Sigmoid()(self.var_param @ matrix_a)
        return matrix_g

    def string(self):
        """
        Just like any class in Python, you can also define custom method on PyTorch modules
        """
        return f'var_param = {self.var_param.item()}'


class RigaDet(nn.Module):
    def __init__(self, weights_size):
        super().__init__()
        self.fc1 = nn.Linear(weights_size, 100, bias=False)
        self.fc2 = nn.Linear(100, 1, bias=False)
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


class RigaExt(nn.Module):
    def __init__(self, config, weights_size):
        super().__init__()
        self.fc1 = nn.Linear(weights_size, 100, bias=False)

        self.fc2 = nn.Linear(100, config["watermark_size"], bias=False)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.th = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.sig(out)
        out = self.fc2(out)
        out = self.sig(out)
        return out


class EncResistant(nn.Module):
    def __init__(self, config, weight_size):
        super().__init__()

        self.fc1 = nn.Linear(weight_size, 100, bias=True)
        self.fc2 = nn.Linear(100, config["expansion_factor"] * weight_size, bias=True)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.matrix_a = 1. * torch.randn(weight_size * config["expansion_factor"], config["watermark_size"],
                                         requires_grad=False).cuda()


    def forward(self, theta_f):
        out = self.fc1(theta_f)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.sig(out @ self.matrix_a)
        return out


class ProjMod(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1_input_size = config[
            "n_features"]  # Adjust based on conv1 output, assuming stride=1 and padding that keeps the length the same
        self.fc1 = nn.Linear(in_features=self.fc1_input_size, out_features=512, bias=True)
        self.fc2 = nn.Linear(in_features=512, out_features=config["watermark_size"], bias=True)
        # self.fc3 = nn.Linear(in_features=config["watermark_size"], out_features=config["watermark_size"])
        self.sig = nn.Sigmoid()


    def forward(self, x):

        out = self.fc1(x)
        out = self.sig(out)
        out = self.fc2(out)
        out = self.sig(out)

        return out


