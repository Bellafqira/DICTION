import torch
import numpy as np
from torch.nn import BCELoss, NLLLoss, BCEWithLogitsLoss


class Metric:

    @staticmethod
    def bit_error_rate(seq_1, seq_2):
        ber_val = [s1 != s2 for s1, s2 in zip(seq_1, seq_2)]
        return 100*sum(ber_val) / len(ber_val)

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
        return 1-(np.array(key1) == np.array(key2)).mean()

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
