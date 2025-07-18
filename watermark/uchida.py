from copy import deepcopy

import torch
from torch import optim, nn
from torch.nn import BCELoss
from tqdm import tqdm

from util.util import Random, TrainModel


def embed(init_model, test_loader, train_loader, config) -> object:
    # Generate a random watermark to insert
    watermark = torch.tensor(Random.get_rand_bits(config["watermark_size"], 0., 1.)).cuda()
    # watermark = torch.tensor(watermark).reshape(1, config["watermark_size"])

    # Instance the target model and Uchida perceptron
    model = deepcopy(init_model)
    print("the names of the layers of the target model")
    for name, param in model.named_parameters():
        print(name)
    weights_selected_layer = [param for name, param in model.named_parameters() if name == config["layer_name"]][0]
    selected_weights = torch.flatten(weights_selected_layer.mean(dim=0))

    # model_uchida = Uchida(selected_weights)
    print("Selected weights shape ", selected_weights.shape)
    # Generate matrix matrix_a
    matrix_a = 1. * torch.randn(len(selected_weights), config["watermark_size"],
                                       requires_grad=False).cuda()
    # Loss and optimizer
    criterion = config["criterion"]

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["wd"])

    scheduler = TrainModel.get_scheduler(optimizer, config)
    ber_ = l_wat = l_global = 1
    matrix_g = 0

    for epoch in range(config["epochs"]):
        train_loss = correct = total = 0
        loop = tqdm(train_loader, leave=True)
        for batch_idx, (inputs, targets) in enumerate(loop):
            # Move tensors to the configured device
            inputs, targets = inputs.to(config["device"]), targets.to(config["device"])
            optimizer.zero_grad()
            outputs = model(inputs)

            # Compute the loss
            weights_selected_layer = [param for name, param in model.named_parameters() if name == config["layer_name"]][0]
            selected_weights = torch.flatten(weights_selected_layer.mean(dim=0))

            matrix_g = torch.nn.Sigmoid()(selected_weights @ matrix_a)

            # sanity check
            assert BCELoss(reduction='sum')(matrix_g, watermark).requires_grad is True, 'broken computational graph :/'
            # λ, control the trade of between WM embedding and Training
            l_main_task = criterion(outputs, targets)
            l_wat = BCELoss(reduction='sum')(matrix_g, watermark)
            l_global = l_main_task + config["lambda_1"] * l_wat

            train_loss += l_main_task.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            assert l_global.requires_grad is True, 'broken computational graph :/'
            # Backpropagation and optimization
            l_global.backward(retain_graph=True)
            optimizer.step()

            ber = _get_ber(matrix_g, watermark)
            _, ber_ = _extract(model, matrix_a, watermark, config["layer_name"])

            loop.set_description(f"Epoch [{epoch}/{config['epochs']}]")
            loop.set_postfix(loss=train_loss / (batch_idx + 1), acc=100. * correct / total,
                             correct_total=f"[{correct}"
                                           f"/{total}]",  l_wat=f"{l_wat:1.4f}",
                             ber=f"{ber:1.3f}",
                             ber_=f"{ber_:1.3f}")
        scheduler.step()

        if (epoch + 1) % config["epoch_check"] == 0:
            with torch.no_grad():
                ber = _get_ber(matrix_g, watermark)
                _, ber_ = _extract(model, matrix_a, watermark, config["layer_name"])
                acc = TrainModel.evaluate(model, test_loader, config)
                print(
                    f"epoch:{epoch}---l_global: {l_global.item():1.3f}---ber: {ber:1.3f}---ber_mean: {ber_:1.3f}"
                    f"--l_wat: {l_wat:1.4f}---acc: {acc}")

            if ber_ == 0:
                print("saving!")
                supplementary = {'model': model, 'matrix_a': matrix_a, 'watermark': watermark, 'ber': ber,
                                 "layer_name": config["layer_name"]}

                TrainModel.save_model(deepcopy(model), acc, epoch, config['save_path'], supplementary)
                print("model saved!")
                break

    return model, ber_


def extract(model_watermarked, supp):
    return _extract(model_watermarked, supp["matrix_a"], supp["watermark"], supp["layer_name"])


def _extract(model, matrix_a, watermark, layer_name):
    weights_selected_layer = [param for name, param in model.named_parameters() if name == layer_name][0]
    w = torch.flatten(weights_selected_layer.mean(dim=0))
    g_ext = torch.nn.Sigmoid()(w @ matrix_a)
    wat_ext = (g_ext > 0.5) * 1.
    ber = _get_ber(wat_ext, watermark)
    return wat_ext, ber


def _get_ber(wat_ext, watermark):
    b_ext = (wat_ext > 0.5) * 1.
    ber = abs(b_ext - watermark).mean()
    return ber.item()
