from copy import deepcopy

import torch
from torch import optim, nn
from tqdm import tqdm

from networks.linear_mod import EncResistant
from util.metric import Metric
from util.util import Random, TrainModel


def embed(init_model, test_loader, train_loader, config) -> object:
    # Generate the watermark to embed
    watermark = Random.get_rand_bits(config["watermark_size"], 0., 1.)
    watermark = torch.tensor(watermark).reshape(1, config["watermark_size"]).cuda()

    # Generate a random watermark
    watermark_rd = Random.get_rand_bits(config["watermark_size"], 1., 0.)
    watermark_rd = torch.tensor(watermark_rd).reshape(1, config["watermark_size"]).cuda()

    # Instance the target and the MappingNet model and move them to the device
    model = deepcopy(init_model)
    weights_selected_layer = [param for name, param in model.named_parameters() if name == config["layer_name"]]
    theta_f = torch.flatten(weights_selected_layer[0].mean(0))
    weight_size = len(theta_f)
    mapping_net = EncResistant(config["expansion_factor"], weight_size).cuda()

    init_model.eval()
    weights_selected_layer_init = [param for name, param in init_model.named_parameters() if
                                   name == config["layer_name"]]
    theta_fn = torch.flatten(weights_selected_layer_init[0].mean(0))
    theta_fn = theta_fn.unsqueeze(0)

    print("theta_f: ", weight_size)

    # Generate matrix matrix_a
    matrix_a = Random.generate_secret_matrix(len(theta_f) * config["expansion_factor"], config["watermark_size"]).cuda()

    # Loss and optimizer
    criterion = config["criterion"]
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': config["lr"], 'weight_decay': config["wd"]},
        {'params': mapping_net.parameters(), 'lr': 1e-4, 'weight_decay': 0}
    ], lr=config["lr"])

    ber_ = l_global = l_wat_orig = l_l1_w = l_wat = 1

    for epoch in range(config["epochs"]):
        train_loss = correct = total = 0
        loop = tqdm(train_loader, leave=True)
        for batch_idx, (x_train, y_train) in enumerate(loop):
            # Move tensors to the configured device
            x_train, y_train = x_train.to(config["device"]), y_train.to(config["device"])
            optimizer.zero_grad()
            y_pred = model(x_train)

            # Compute the l_global
            weights_selected_layer = [param for name, param in model.named_parameters() if name == config["layer_name"]]
            theta_f = torch.flatten(weights_selected_layer[0].mean(0)).cuda()
            theta_f = theta_f.unsqueeze(0)

            matrix_g = mapping_net(theta_f.cuda(), matrix_a.cuda())
            matrix_gn = mapping_net(theta_fn.cuda(), matrix_a.cuda())

            # sanity check
            assert Metric.bce_(matrix_g, watermark).requires_grad is True, 'broken computational graph :/'

            l_main_task = criterion(y_pred, y_train)
            l_wat = Metric.bce_(matrix_g, watermark)
            l_wat_orig = Metric.bce_(matrix_gn, watermark_rd)
            l_l1_w = torch.cdist(theta_f, torch.zeros_like(theta_f), p=1)
            l_global = (l_main_task + config["lambda_1"] * l_wat + config["lambda_2"] * l_wat_orig +
                        config["lambda_3"] * l_l1_w)

            # Backpropagation and optimization
            l_global.backward(retain_graph=True)
            optimizer.step()

            train_loss += l_main_task.item()
            _, predicted = y_pred.max(1)
            total += y_train.size(0)
            correct += predicted.eq(y_train).sum().item()

            assert l_global.requires_grad is True, 'broken computational graph :/'

            _, ber_ = _extract(model, matrix_a, watermark, config["layer_name"], mapping_net)

            loop.set_description(f"Epoch [{epoch}/{config['epochs']}]")
            loop.set_postfix(l_main_task=train_loss / (batch_idx + 1), acc=100. * correct / total,
                             correct_total=f"[{correct}"
                                           f"/{total}]", loss_w=f"{l_wat:1.4f}", loss_w2=f"{l_wat_orig:1.4f}",
                             loss_bs=f"{l_l1_w.item():1.4f}",
                             ber_=f"{ber_:1.3f}")
        # scheduler.step()

        if (epoch + 1) % config["epoch_check"] == 0:
            with torch.no_grad():
                _, ber_ = _extract(model, matrix_a, watermark, config["layer_name"], mapping_net)
                acc = TrainModel.evaluate(model, test_loader, config)
                print(
                    f"epoch:{epoch}---l_global: {l_global.item():1.3f}---ber: {ber_:1.3f}---l_wat: "
                    f"{l_wat:1.4f}---loss_w2: {l_wat_orig:.3f}---l_l1_w: {l_l1_w.item():.3f}---acc: {acc}")

            if ber_ == 0:
                print("saving... watermarked model! ")
                supplementary = {'model': model, 'matrix_a': matrix_a, 'watermark': watermark, 'ber': ber_,
                                 "layer_name": config["layer_name"], "mapping_net": mapping_net}

                TrainModel.save_model(deepcopy(model), acc, epoch, config['save_path'], supplementary)
                print("model saved!")
                break
    return model, ber_


def extract(model_watermarked, supp):
    return _extract(model_watermarked, supp["matrix_a"], supp["watermark"], supp["layer_name"],
                    supp["mapping_net"])


def _extract(model, matrix_a, watermark, layer_name, mapping_net):
    weights_selected_layer = [param for name, param in model.named_parameters() if name == layer_name]
    theta_f = torch.flatten(weights_selected_layer[0].mean(0))
    g_ext = mapping_net(theta_f, matrix_a)
    wat_ext = 1. * (g_ext > 0.5)
    ber = _get_ber(wat_ext, watermark)
    return wat_ext, ber.item()


def _get_ber(wat_ext, watermark):
    ber = abs(wat_ext - watermark).mean()
    return ber
