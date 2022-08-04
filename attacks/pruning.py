import torch.nn.utils.prune as prune
import torch


def pruning(model, config_attack):
    """Pruning function with pytorch
    :param config_attack:
    :argument model to prune
    :return pruned model
    """
    parents = [module for module in model.children()]
    leaves = []
    while parents:
        m = parents.pop(0)
        if not list(m.children()):
            leaves.append(m)
        else:
            for x in m.children():
                if not list(x.children()):
                    leaves.append(x)
                else:
                    parents.append(x)

    parameters_to_prune = [(module, 'weight') for module in leaves \
                           if (isinstance(module, torch.nn.modules.conv.Conv2d) or isinstance(module,
                                                                                              torch.nn.modules.Linear))]
    print("number of pruned layers", len(parameters_to_prune))
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=config_attack["amount"])

    for module in leaves:
        if (isinstance(module, torch.nn.modules.conv.Conv2d) or isinstance(module,
                                                                           torch.nn.modules.Linear)):
            prune.remove(module, 'weight')

    sum_weights = 0
    sum_nelemnt = 0
    for name, param in model.named_parameters():
        if config_attack["architecture"] == "ResNet18":
            if 'conv' in name or "linear.w" in name:
                print(name)
                sum_weights += float(torch.sum(param == 0))
                sum_nelemnt += float(param.nelement())
        elif config_attack["architecture"] == "MLP" or config_attack["architecture"] == "CNN":
            if 'weight' in name:
                sum_weights += float(torch.sum(param == 0))
                sum_nelemnt += float(param.nelement())
        else:
            raise Exception("Architecture doesnt exist")

    print("Sparsity in the model: {:.2f}%".format(100. * sum_weights/sum_nelemnt))

    return model


def print_sparsity(model):
    """show the sparsity of a model
    :argument model
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
