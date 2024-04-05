import torch
from torch.nn.utils import prune
from copy import deepcopy


def pruning(init_model, amount):
    """
    Prune the given model's parameters, including weights and biases across all layers
    that support these parameters, such as Conv2d, Linear, and BatchNorm layers.

    Args:
        init_model (torch.nn.Module): The model to prune.
        amount (double): pruning parameters such as "amount".

    Returns:
        torch.nn.Module: The pruned model.
    """
    # Create a deep copy of the model to keep the original model unchanged.
    model = deepcopy(init_model)

    # Apply pruning to all layers with 'weight' and 'bias' parameters.
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')  # Make pruning permanent
        if hasattr(module, 'bias') and module.bias is not None:
            prune.l1_unstructured(module, name='bias', amount=amount)
            prune.remove(module, 'bias')  # Make pruning permanent

    # Calculate and print the sparsity of the model.
    sum_weights, sum_elements = 0, 0
    for _, param in model.named_parameters():
        sum_weights += float(torch.sum(param == 0))
        sum_elements += float(param.nelement())

    sparsity = 100. * sum_weights / sum_elements
    print(f"Sparsity in the model: {sparsity:.2f}%")

    return model
