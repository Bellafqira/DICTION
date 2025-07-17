import torch
import torch.nn as nn
import copy
from typing import Tuple

# This code present the attack proposed in "Rethinking White-Box Watermarks on Deep Learning Models
# under Neural Structural Obfuscation" : https://www.usenix.org/system/files/sec23fall-prepub-444-yan-yifan.pdf

def neuron_clique(model: nn.Module, layer_name: str, num_dummy: int = 2) -> nn.Module:
    if num_dummy < 2:
        raise ValueError("NeuronClique needs at least 2 dummy neurons to cancel out")

    model = copy.deepcopy(model)

    # Get the target layer by name
    try:
        target_layer = model.get_submodule(layer_name)
    except AttributeError:
        raise ValueError(f"No layer named '{layer_name}' found in model")

    if not isinstance(target_layer, nn.Linear):
        raise ValueError(f"Layer '{layer_name}' must be of type nn.Linear")

    # Find the parent module and attribute name
    parent_module = model
    modules = layer_name.split('.')
    for name in modules[:-1]:
        parent_module = getattr(parent_module, name)
    attr_name = modules[-1]

    # Find the next Linear layer (after layer_name in model.named_modules())
    found = False
    next_linear_name = None
    for name, layer in model.named_modules():
        if name == layer_name:
            found = True
            continue
        if found and isinstance(layer, nn.Linear):
            next_linear_name = name
            break

    if next_linear_name is None:
        raise ValueError("No Linear layer found after the target layer")

    next_layer = model.get_submodule(next_linear_name)

    # === Expand target layer ===
    old_weight = target_layer.weight.data
    old_bias = target_layer.bias.data if target_layer.bias is not None else None

    new_target = nn.Linear(target_layer.in_features,
                           target_layer.out_features + num_dummy,
                           bias=target_layer.bias is not None)

    new_target.weight.data[:target_layer.out_features, :] = old_weight
    if old_bias is not None:
        new_target.bias.data[:target_layer.out_features] = old_bias
        new_target.bias.data[target_layer.out_features:] = torch.zeros(num_dummy)

    shared_input_weights = torch.randn(1, target_layer.in_features)
    for i in range(num_dummy):
        new_target.weight.data[target_layer.out_features + i, :] = shared_input_weights[0, :]

    # === Expand next layer ===
    old_next_weight = next_layer.weight.data
    old_next_bias = next_layer.bias.data if next_layer.bias is not None else None

    new_next = nn.Linear(next_layer.in_features + num_dummy,
                         next_layer.out_features,
                         bias=next_layer.bias is not None)

    new_next.weight.data[:, :next_layer.in_features] = old_next_weight
    if old_next_bias is not None:
        new_next.bias.data = old_next_bias

    dummy_output_weights = torch.randn(next_layer.out_features, num_dummy - 1)
    last_weight = -dummy_output_weights.sum(dim=1, keepdim=True)
    dummy_output_weights = torch.cat([dummy_output_weights, last_weight], dim=1)

    new_next.weight.data[:, next_layer.in_features:] = dummy_output_weights

    # === Replace layers in the model ===
    setattr(parent_module, attr_name, new_target)

    # Also replace the next layer
    next_parent = model
    next_modules = next_linear_name.split('.')
    for name in next_modules[:-1]:
        next_parent = getattr(next_parent, name)
    next_attr = next_modules[-1]
    setattr(next_parent, next_attr, new_next)

    return model

def neuron_split(model: nn.Module, layer_name: str, neuron_idx: int, num_splits: int = 2) -> nn.Module:
    if num_splits < 2:
        raise ValueError("NeuronSplit needs at least 2 neurons")

    model = copy.deepcopy(model)

    # Get the target layer by name
    try:
        target_layer = model.get_submodule(layer_name)
    except AttributeError:
        raise ValueError(f"No layer named '{layer_name}' found in model")

    if not isinstance(target_layer, nn.Linear):
        raise ValueError(f"Layer '{layer_name}' must be of type nn.Linear")

    if neuron_idx >= target_layer.out_features:
        raise ValueError(f"Neuron index {neuron_idx} out of range")

    # Find the parent module and attribute name
    parent_module = model
    modules = layer_name.split('.')
    for name in modules[:-1]:
        parent_module = getattr(parent_module, name)
    attr_name = modules[-1]

    # Find the next Linear layer
    found = False
    next_linear_name = None
    for name, layer in model.named_modules():
        if name == layer_name:
            found = True
            continue
        if found and isinstance(layer, nn.Linear):
            next_linear_name = name
            break

    if next_linear_name is None:
        raise ValueError("No Linear layer found after the target layer")

    next_layer = model.get_submodule(next_linear_name)

    # === Expand target layer ===
    old_weight = target_layer.weight.data
    old_bias = target_layer.bias.data if target_layer.bias is not None else None

    target_input_weight = old_weight[neuron_idx, :]
    target_bias = old_bias[neuron_idx] if old_bias is not None else 0

    new_target = nn.Linear(target_layer.in_features,
                           target_layer.out_features + (num_splits - 1),
                           bias=target_layer.bias is not None)

    # Copy weights before the neuron
    new_target.weight.data[:neuron_idx, :] = old_weight[:neuron_idx, :]
    # Copy weights after the neuron
    new_target.weight.data[neuron_idx + num_splits:, :] = old_weight[neuron_idx + 1:, :]

    # Copy biases before and after
    if old_bias is not None:
        new_target.bias.data[:neuron_idx] = old_bias[:neuron_idx]
        new_target.bias.data[neuron_idx + num_splits:] = old_bias[neuron_idx + 1:]

    # Replace target neuron with num_splits identical neurons
    for i in range(num_splits):
        new_idx = neuron_idx + i
        new_target.weight.data[new_idx, :] = target_input_weight
        if old_bias is not None:
            new_target.bias.data[new_idx] = target_bias

    # === Expand next layer ===
    old_next_weight = next_layer.weight.data
    old_next_bias = next_layer.bias.data if next_layer.bias is not None else None

    original_output_weights = old_next_weight[:, neuron_idx]  # (out_features,)

    new_next = nn.Linear(next_layer.in_features + (num_splits - 1),
                         next_layer.out_features,
                         bias=next_layer.bias is not None)

    # Copy weights before and after the neuron
    new_next.weight.data[:, :neuron_idx] = old_next_weight[:, :neuron_idx]
    new_next.weight.data[:, neuron_idx + num_splits:] = old_next_weight[:, neuron_idx + 1:]

    if old_next_bias is not None:
        new_next.bias.data = old_next_bias

    # Distribute output weights among the split neurons
    split_weights = torch.randn(next_layer.out_features, num_splits - 1).to('cuda')
    last_weight = original_output_weights.unsqueeze(1).to('cuda') - split_weights.sum(dim=1, keepdim=True).to('cuda')
    all_weights = torch.cat([split_weights, last_weight], dim=1)

    for i in range(num_splits):
        new_next.weight.data[:, neuron_idx + i] = all_weights[:, i]

    # === Replace layers ===  +
    setattr(parent_module, attr_name, new_target)

    next_parent = model
    next_modules = next_linear_name.split('.')
    for name in next_modules[:-1]:
        next_parent = getattr(next_parent, name)
    next_attr = next_modules[-1]
    setattr(next_parent, next_attr, new_next)

    return model