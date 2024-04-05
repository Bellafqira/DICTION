from copy import deepcopy

import torch
from torch import optim, nn
from tqdm import tqdm

from networks.linear_mod import Uchida, RigaExt, RigaDet
from util.metric import Metric
from util.util import Random, TrainModel


def embed(init_model, test_loader, train_loader, config) -> object:
    # Generate a random watermark to insert
    watermark = torch.tensor(Random.get_rand_bits(config["watermark_size"], 0., 1.)).cuda()
    # watermark = torch.tensor(watermark).reshape(1, config["watermark_size"])

    rd_watermark = torch.tensor(Random.get_rand_bits(config["watermark_size"], 1., 0.)).cuda()
    # rd_watermark = torch.tensor(rd_watermark).reshape(1, config["watermark_size"]).cuda()

    # Instance the target model and Uchida perceptron
    model = deepcopy(init_model)
    weights_selected_layer = [param for name, param in model.named_parameters() if name == config["layer_name"]]
    init_w = torch.flatten(weights_selected_layer[0].mean(0)).clone()

    model_ext = RigaExt(config, len(init_w))
    model_det = RigaDet(len(init_w))

    model_det = model_det.cuda()
    model_ext = model_ext.cuda()
    print("init_w.shape = ", init_w.shape)

    # Loss and optimizer
    criterion = config["criterion"]
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': config["lr"], 'weight_decay': config["wd"]},
        {'params': model_ext.parameters(), 'lr': config["lr"], 'weight_decay': 0, 'betas': (0.5, 0.999)}

    ], lr=config["lr"])

    optimizer_det = optim.Adam(model_det.parameters(), lr=1e-3, betas=(0.5, 0.999))

    for epoch in range(config["epochs"]):
        train_loss = correct = total = 0
        loop = tqdm(train_loader, leave=True)
        for batch_idx, (inputs, targets) in enumerate(loop):

            # Compute the loss
            weights_selected_layer = [param for name, param in model.named_parameters() if name == config["layer_name"]]
            w = torch.flatten(weights_selected_layer[0].mean(0))

            w_sorted = torch.sort(w.detach())[0]
            init_w_sorted = torch.sort(init_w.detach())[0]
            out_detector_wat = model_det(w_sorted)
            out_detector_non = model_det(init_w_sorted)
            # sanity check

            optimizer_det.zero_grad()
            loss_det_non = Metric.bce_(out_detector_non, torch.ones(1).cuda())
            loss_det_wat = Metric.bce_(out_detector_wat, torch.zeros(1).cuda())

            loss_det = loss_det_non + loss_det_wat  # torch.log(out_detector_non) + torch.log(1 - out_detector_wat)

            assert loss_det.requires_grad is True, 'broken computational graph :/'
            loss_det.backward(retain_graph=True)
            optimizer_det.step()
            with torch.no_grad():
                for param in model_det.parameters():
                    param.clamp_(-0.01, 0.01)

            # Move tensors to the configured device
            inputs, targets = inputs.to(config["device"]), targets.to(config["device"])
            outputs = model(inputs)

            out_watermark = model_ext(w)
            init_out_watermark = model_ext(init_w.detach())

            optimizer.zero_grad()
            loss_0 = criterion(outputs, targets)
            loss_1_1 = Metric.bce_(out_watermark, watermark)
            loss_1_2 = Metric.bce_(init_out_watermark, rd_watermark)
            loss_1_3 = Metric.bce_(out_detector_wat.detach(), torch.ones(1).cuda())
            loss = loss_0 + config["lambda_1"] * (loss_1_1 + loss_1_2) - config["lambda_2"] * loss_1_3

            train_loss += loss_0.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            assert loss.requires_grad is True, 'broken computational graph :/'
            # Backpropagation and optimization
            loss.backward(retain_graph=True)
            optimizer.step()

            _, ber = _get_ber(out_watermark.cpu().detach().numpy(), watermark.cpu())
            _, ber_ = _extract(model, model_ext, watermark, config["layer_name"], print_ber=False)

            loop.set_description(f"Epoch [{epoch}/{config['epochs']}]")
            loop.set_postfix(loss=train_loss / (batch_idx + 1), acc=100. * correct / total,
                             correct_total=f"[{correct}"
                                           f"/{total}]", loss_ext=f"{loss.item():1.4f}", loss_det=f"{loss_det.item():1.4f}",
                             ber=f"{ber:1.3f}",
                             ber_=f"{ber_:1.3f}")
        # scheduler.step()

        if (epoch + 1) % config["epoch_check"] == 0:
            with torch.no_grad():
                _, ber = _get_ber(out_watermark.cpu().detach().numpy(), watermark.cpu())
                _, ber_ = _extract(model, model_ext, watermark, config["layer_name"], print_ber=False)

                acc = TrainModel.evaluate(model, test_loader, config)
                print(
                    f"epoch:{epoch}---loss: {loss.item():1.3f}---ber: {ber:1.3f}---ber_mean: {ber_:1.3f}---loss1: "
                    f"{loss_0.item():1.4f}---loss2: {loss_det.item():.3f}---acc: {acc}")

        if ber_ == 0 and epoch >= config["epoch_check"]:
            print("saving!")
            supplementary = {'model': model, 'model_ext': model_ext, 'watermark': watermark, 'ber': ber,
                             "layer_name": config["layer_name"]}

            TrainModel.save_model(deepcopy(model), acc, epoch, config['save_path'], supplementary)
            break
    return model, ber_


def extract(model_watermarked, supp):
    return _extract(model_watermarked, supp["model_ext"], supp["watermark"], supp["layer_name"])


def _extract(model, model_det, watermark, layer_name, print_ber=True):
    weights_selected_layer = [param for name, param in model.named_parameters() if name == layer_name]
    w = torch.flatten(weights_selected_layer[0].mean(0))
    watermark_out = model_det(w)
    b_ext, ber = _get_ber(watermark_out.cpu().detach().numpy(), watermark.cpu())
    if print_ber:
        print(f'BER after extraction = {ber}')
    return b_ext, ber


def _get_ber(wat_out, watermark):
    b_ext = 1. * (wat_out > 0.5)
    ber = Metric.get_ber(b_ext, watermark)
    return b_ext, ber
