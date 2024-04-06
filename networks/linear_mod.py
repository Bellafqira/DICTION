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


class EncResistant(nn.Module):
    def __init__(self, expansion_factor, weight_size):
        super().__init__()
        self.fc1 = nn.Linear(weight_size, 100, bias=False)
        self.fc2 = nn.Linear(100, expansion_factor * weight_size, bias=False)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, theta_f, matrix_a):
        out = self.fc1(theta_f)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.sig(out @ matrix_a)
        return out


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


class LinearMod(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config["n_features"], 128)
        self.fc2 = nn.Linear(128, config["watermark_size"])
        self.fc3 = nn.Linear(config["watermark_size"], config["watermark_size"])

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
