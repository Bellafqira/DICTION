import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import BCELoss, NLLLoss, BCEWithLogitsLoss


class Metric:

    @staticmethod
    def bit_error_rate(seq_1, seq_2):
        ber_val = [s1 != s2 for s1, s2 in zip(seq_1, seq_2)]
        return 100 * sum(ber_val) / len(ber_val)

    @staticmethod
    def norm_computation(sw_0):
        mse = np.linalg.norm(sw_0) / np.sqrt(len(sw_0))
        return mse

    @staticmethod
    def bce(a, b):
        "binary cross entropy"
        return BCEWithLogitsLoss(reduction='sum')(a, b)

    @staticmethod
    def bce_(matrix_g, b):
        "binary cross entropy"
        return BCELoss(reduction='mean')(matrix_g, b)

    @staticmethod
    def get_ber(key1, key2):
        assert key1.shape == key2.shape, "Tensors must be the same shape"
        return 1 - (np.array(key1) == np.array(key2)).mean()

    @staticmethod
    def mse(a, b):
        """Quadratic error"""
        return torch.nn.MSELoss(reduction='mean')(a, b)

    @staticmethod
    def binary_acc(y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))
        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum / y_test.shape[0]
        acc = torch.round(acc * 100)

        return acc

    @staticmethod
    def coupling_regularization(model1, model2, lambda_coupling=1):
        coupling_loss = 0.0
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            coupling_loss += torch.norm(param1 - param2, p=2)
            # Use L2 norm to calculate distance between parameters

        coupling_loss *= lambda_coupling   # Weighting factor to control the importance of regularization

        return coupling_loss

class SignLoss(nn.Module):
    def __init__(self, alpha, b=None):
        super(SignLoss, self).__init__()
        self.alpha = alpha
        self.register_buffer('b', b)
        self.loss = 0
        self.acc = 0
        self.scale_cache = None

    def set_b(self, b):
        self.b.copy_(b)

    def get_acc(self):
        if self.scale_cache is not None:
            acc = (torch.sign(self.b.view(-1)) == torch.sign(self.scale_cache.view(-1))).float().mean()
            return acc
        else:
            raise Exception('scale_cache is None')

    def get_loss(self):
        if self.scale_cache is not None:
            loss = (self.alpha * F.relu(-self.b.view(-1) * self.scale_cache.view(-1) + 0.1)).sum()
            return loss
        else:
            raise Exception('scale_cache is None')

    def add(self, scale):
        self.scale_cache = scale

        # hinge loss concept
        # f(x) = max(x + 0.5, 0)*-b
        # f(x) = max(x + 0.5, 0) if b = -1
        # f(x) = max(0.5 - x, 0) if b = 1

        # case b = -1
        # - (-1) * 1 = 1 === bad
        # - (-1) * -1 = -1 -> 0 === good

        # - (-1) * 0.6 + 0.5 = 1.1 === bad
        # - (-1) * -0.6 + 0.5 = -0.1 -> 0 === good

        # case b = 1
        # - (1) * -1 = 1 -> 1 === bad
        # - (1) * 1 = -1 -> 0 === good

        # let it has minimum of 0.1
        self.loss += self.get_loss()
        self.loss += (0.00001 * scale.view(-1).pow(2).sum())  # to regularize the scale not to be so large
        self.acc += self.get_acc()

    def reset(self):
        self.loss = 0
        self.acc = 0
        self.scale_cache = None

    # def to(self, *args, **kwargs):
    #     self.loss = self.loss.to(args[0])
    #     return super().to(*args, **kwargs)
