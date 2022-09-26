import torch.nn.utils.prune as prune
import torch
from copy import deepcopy


def pruning(model_init, config_attack):
    """Pruning function with pytorch
    :param model_init: model to prune
    :param config_attack:
    :return pruned model
    """
    model = deepcopy(model_init)

    for module in model.modules():
        if isinstance(module, torch.nn.modules.conv.Conv2d) or isinstance(module, torch.nn.modules.Linear):
            prune.l1_unstructured(module, 'weight', config_attack["amount"])
            prune.remove(module, 'weight')

    sum_weights = 0
    sum_nelemnt = 0
    for name, param in model.named_parameters():
        sum_weights += float(torch.sum(param == 0))
        sum_nelemnt += float(param.nelement())

    print("Sparsity in the model: {:.2f}%".format(100. * sum_weights / sum_nelemnt))

    return model


def print_sparsity(model):
    """show the sparsity of a model
    :param model
    :return print the sparsity of the model
    """
    print("Sparsity in fc1.weight: {:.2f}%".format(100. * float(torch.sum(model.fc1.weight == 0)) /
                                                   float(model.fc1.weight.nelement())
                                                   )
          )
    print(
        "Sparsity in fc2.weight: {:.2f}%".format(
            100. * float(torch.sum(model.fc2.weight == 0))
            / float(model.fc2.weight.nelement())
        )
    )

    print(
        "Global sparsity: {:.2f}%".format(
            100. * float(
                torch.sum(model.fc1.weight == 0)
                + torch.sum(model.fc2.weight == 0)
            )
            / float(
                model.fc1.weight.nelement()
                + model.fc2.weight.nelement()
            )
        )
    )
