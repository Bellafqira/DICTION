from copy import deepcopy

import torch
from torch import optim, nn
from tqdm import tqdm

from networks.linear_mod import Uchida
from util.metric import Metric
from util.util import Random, TrainModel


def embed(init_model, test_loader, train_loader, config) -> object:
    # Generate a random watermark to insert
    watermark = torch.tensor(Random.get_rand_bits(config["watermark_size"], 0., 1.)).cuda()
    # watermark = torch.tensor(watermark).reshape(1, config["watermark_size"])

    # Instance the target model and Uchida perceptron
    model = deepcopy(init_model)
    weights_selected_layer = [param for name, param in model.named_parameters() if name == config["layer_name"]]
    w = torch.flatten(weights_selected_layer[0].mean(0))
    model_uchida = Uchida(w)
    print(w.shape)
    # Generate matrix matrix_a
    matrix_a = Random.generate_secret_matrix(len(w), config["watermark_size"]).cuda()

    # Loss and optimizer
    criterion = config["criterion"]
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': config["lr"], 'weight_decay': config["wd"]},
        {'params': model_uchida.parameters(), 'lr': 1e-4, 'weight_decay': 0}
    ], lr=config["lr"])

    for epoch in range(config["epochs"]):
        train_loss = correct = total = 0
        loop = tqdm(train_loader, leave=True)
        for batch_idx, (inputs, targets) in enumerate(loop):
            # Move tensors to the configured device
            inputs, targets = inputs.to(config["device"]), targets.to(config["device"])
            optimizer.zero_grad()
            outputs = model(inputs)
            # Compute the loss
            # weights_selected_layer = [param for param in model.parameters()][config_embed["index_layer"]]
            weights_selected_layer = [param for name, param in model.named_parameters() if name == config["layer_name"]]
            w = torch.flatten(weights_selected_layer[0].mean(0))

            matrix_g = model_uchida(matrix_a)  # <----------- Uchida Method

            # sanity check
            assert Metric.bce_(matrix_g, watermark).requires_grad is True, 'broken computational graph :/'
            # λ, control the trade of between WM embedding and Training
            loss0 = criterion(outputs, targets)
            loss1 = Metric.bce_(matrix_g, watermark)
            loss2 = Metric.mse(w, model_uchida.var_param)
            loss = 0.1*loss0 + config["lambda_1"] * loss1 + config["lambda_2"] * loss2

            train_loss += loss0.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            assert loss.requires_grad is True, 'broken computational graph :/'
            # Backpropagation and optimization
            loss.backward(retain_graph=True)
            optimizer.step()

            ber = _get_ber(matrix_g, watermark)
            _, ber_ = _extract(model, matrix_a, watermark, config["layer_name"], print_ber=False)

            loop.set_description(f"Epoch [{epoch}/{config['epochs']}]")
            loop.set_postfix(loss=train_loss / (batch_idx + 1), acc=100. * correct / total,
                             correct_total=f"[{correct}"
                                           f"/{total}]", loss2=f"{loss2:1.4f}", loss1=f"{loss1:1.4f}",
                             ber=f"{ber:1.3f}",
                             ber_=f"{ber_:1.3f}")
        # scheduler.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                ber = _get_ber(matrix_g, watermark)

                _, ber_ = _extract(model, matrix_a, watermark, config["layer_name"], print_ber=False)

                acc = TrainModel.evaluate(model, test_loader, config)
                print(
                    f"epoch:{epoch}---loss: {loss.item():1.3f}---ber: {ber:1.3f}---ber_mean: {ber_:1.3f}---loss1: "
                    f"{loss1:1.4f}---loss2: {loss2:.3f}---acc: {acc}")

        if ber_ == 0 and epoch >= config["epoch_check"]:
            print("saving!")
            supplementary = {'model': model, 'matrix_a': matrix_a, 'watermark': watermark, 'ber': ber,
                             "layer_name": config["layer_name"]}

            TrainModel.save_model(deepcopy(model), acc, epoch, config['save_path'], supplementary)
            break
    return model, ber_


def extract(model_watermarked, supp):
    return _extract(model_watermarked, supp["matrix_a"], supp["watermark"], supp["layer_name"])


def _extract(model, matrix_a, watermark, layer_name, print_ber=True):
    weights_selected_layer = [param for name, param in model.named_parameters() if name == layer_name]
    w = torch.flatten(weights_selected_layer[0].mean(0))

    g_ext = torch.nn.Sigmoid()(w @ matrix_a)
    b_ext = (g_ext > 0.5) * 1

    ber = _get_ber(b_ext, watermark)
    if print_ber:
        print(f'BER after extraction = {ber}')
    return b_ext, ber


def _get_ber(matrix_g, watermark):
    b_ext = (matrix_g > 0.5) * 1.
    ber = abs(b_ext - watermark).mean()
    return ber
