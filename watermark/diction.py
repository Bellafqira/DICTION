from copy import deepcopy
import random
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from tqdm import tqdm
from networks.linear_mod import LinearMod
from util.metric import Metric

from PIL import Image, ImageDraw, ImageFont

# Check Device configuration
from util.util import TrainModel, Random, AddGaussianNoise, AddWhiteSquareTransform, CustomTensorDataset


def embed(init_model, test_loader, train_loader, config) -> object:
    # Generate the watermark to embed
    watermark = Random.get_rand_bits(config["watermark_size"], 0., 1.)
    watermark = torch.tensor(watermark).reshape(1, config["watermark_size"])

    # Instance the target model
    model = deepcopy(init_model)

    # Get the activation layer of the original model and make sure that its parameters are not trainable
    extractor_orig = create_feature_extractor(init_model, [config["layer_name"]])

    # Get the activation layer of the target model
    extractor_wat = create_feature_extractor(model, [config["layer_name"]])

    # show the graph of the model
    print("the model graph :=> ", get_graph_node_names(init_model)[0])
    init_model.eval()
    for param in extractor_orig.parameters():
        param.requires_grad = False

    # Generate the trigger set based on a Latent space
    x_key, y_key = next(iter(train_loader))
    transform_train = transforms.Compose([
        # transforms.RandomCrop(size=32, padding=4),
        # transforms.RandomHorizontalFlip(),
        AddGaussianNoise(config["mean"], config["std"]),
        AddWhiteSquareTransform(square_size=config["square_size"], start_x=config["start_x"], start_y=config["start_y"]),
    ])
    dataset_key = CustomTensorDataset(x_key, y_key, transform_list=transform_train)
    key_loader = DataLoader(dataset=dataset_key, batch_size=config["batch_size"], shuffle=True)

    # Get the number of features of the layer
    n_features_layer = len(extractor_orig(x_key.cuda())[config["layer_name"]][0].view(-1))
    config["n_features"] = int(n_features_layer * config["n_features"])

    print("n_features_layer: ", n_features_layer, "n_features selected: ", config["n_features"])

    # Instance the linear mod
    linear_mod = LinearMod(config).to(config["device"])

    # Select the indices of the activations maps that will be fed to the projection model
    indices = random.choices(range(n_features_layer), k=config["n_features"])

    # Init the BER and the loss
    ber_ = ber = l_global = l_proj = 1

    # The training parameters
    criterion = config["criterion"]
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': config["lr"], 'weight_decay': config["wd"]},
        {'params': linear_mod.parameters(), 'lr': 1e-2, 'weight_decay': 1e-4 }
    ], lr=config["lr"])
    scheduler = TrainModel.get_scheduler(optimizer, config)

    # Generate a random watermark
    watermark_rd = Random.get_rand_bits(config["watermark_size"], 1., 0.)
    watermark_rd = torch.tensor(watermark_rd).reshape(1, config["watermark_size"])

    # Start the training
    for epoch in range(config["epochs"]):
        embed_loss = train_loss = correct = total = 0
        loop = tqdm(train_loader, leave=True)

        for batch_idx, (x_train, y_train) in enumerate(loop):

            y_train = y_train.type(torch.long)
            x_train, y_train = x_train.to(config["device"]), y_train.to(config["device"])
            optimizer.zero_grad()
            y_pred = model(x_train)

            x_key, _ = next(iter(key_loader))
            x_key = x_key.cuda()

            # Get activation maps of the watermarked model
            act_wat = extractor_wat(x_key)[config["layer_name"]]
            act_wat = act_wat.view(act_wat.shape[0], -1)
            act_wat = act_wat[:, torch.tensor(indices).cuda()]
            # act_wat = torch.mean(act_wat, dim=0, keepdim=True)

            # Get activation maps of the original model
            act_orig = extractor_orig(x_key)[config["layer_name"]]
            act_orig = act_orig.view(act_orig.shape[0], -1)
            act_orig = act_orig[:, torch.tensor(indices).cuda()]
            # act_orig = torch.mean(act_orig, dim=0, keepdim=True)

            # Get the output of the projection model
            # watermark_out = linear_mod(act_wat.repeat(x_key_len, 1))
            # watermark_orig_out = linear_mod(act_orig.repeat(x_key_len, 1))

            watermark_out = linear_mod(act_wat)
            watermark_orig_out = linear_mod(act_orig)

            # Compute the loss
            l_main_task = criterion(y_pred, y_train)
            l_wat = Metric.bce_(watermark_out, watermark.repeat(len(watermark_out), 1).cuda())
            l_wat_orig = Metric.bce_(watermark_orig_out, watermark_rd.repeat(len(watermark_out), 1).cuda())
            l_proj = l_wat + l_wat_orig

            l_global = l_main_task + config["lambda"] * l_proj
                        # + Metric.coupling_regularization(model, init_model, lambda_coupling=1e-5))

            assert l_global.requires_grad == True, 'broken computational graph :/'

            l_global.backward(retain_graph=True)
            optimizer.step()

            train_loss += l_main_task.item()
            embed_loss += l_proj.item()
            _, predicted = y_pred.max(1)
            total += y_train.size(0)
            correct += predicted.eq(y_train).sum().item()

            # update the progress bar
            _, ber = _get_ber(watermark_out.cpu().detach().numpy(), watermark.repeat(len(watermark_out), 1).cpu())
            loop.set_description(f"Epoch [{epoch}/{config['epochs']}]")
            loop.set_postfix(l_main_task=train_loss / (batch_idx + 1), acc=100. * correct / total,
                             correct_total=f"[{correct}"
                                           f"/{total}]", l_proj=f"{embed_loss / (batch_idx + 1):1.4f}",
                             ber=f"{ber:1.3f}", l_wat=f"{l_wat:1.3f}", l_wat_orig=f"{l_wat_orig:1.3f}")
        scheduler.step()

        if (epoch + 1) % config["epoch_check"] == 0:
            with torch.no_grad():
                # Get the trigger set
                x_key, _ = next(iter(key_loader))
                # Get the activation maps of the watermarked model
                act_wat = extractor_wat(x_key.cuda())[config["layer_name"]]
                act_wat = act_wat.view(act_wat.shape[0], -1)
                act_wat = act_wat[:, torch.tensor(indices).cuda()]
                # act_wat = torch.mean(act_wat, dim=0, keepdim=True)

                # get the BER
                _, ber_ = _extract(act_wat, linear_mod, watermark.cuda())
                # get the accuracy of the watermarked model
                acc = TrainModel.evaluate(model, test_loader, config)
                print(
                    f"epoch:{epoch}---l_global: {l_global.item():1.7f}---ber_mean: {ber_:1.3f}"
                    f"---param_var_loss: "
                    f"{l_proj:1.4f}---acc: {acc}")

            # 'x_key': key_loader,
            if ber_ == 0:
                print("saving... watermarked model! ")
                supplementary = {'model': model, 'matrix_a': linear_mod, 'watermark': watermark,
                                 'x_key': x_key, 'y_key': y_key, 'ber': ber,
                                 "layer_name": config["layer_name"], "indices": indices}
                TrainModel.save_model(deepcopy(model), acc, epoch, config['save_path'], supplementary)
                print("model saved!")
                break

    return model, ber_


def extract(model_watermarked, supp):
    model_watermarked.eval()
    extractor = create_feature_extractor(model_watermarked, [supp["layer_name"]])
    # x_key, _ = next(iter(supp["x_key"]))
    x_key = supp["x_key"]
    act = extractor(x_key.cuda())[supp["layer_name"]]
    act = act.view(act.shape[0], -1)
    act = act[:, supp["indices"]]
    # act = torch.mean(act, dim=0, keepdim=True)

    wat_ext, ber = _extract(act.cuda(), supp["matrix_a"], supp["watermark"])
    wat_ext = torch.tensor(wat_ext).mean(dim=0).reshape(1, supp["watermark"].shape[1])
    # ber = Metric.get_ber(wat_ext, supp["watermark"])
    return wat_ext, ber


def _extract(act, model, watermark):
    watermark_out = model(act)
    wat_ext, ber = _get_ber(watermark_out.cpu().detach().numpy(), watermark.repeat(len(watermark_out), 1).cpu())
    return wat_ext, ber


def _get_ber(wat_out: object, watermark: object) -> object:
    wat_ext = 1. * (wat_out > 0.5)
    ber = Metric.get_ber(wat_ext, watermark)
    return wat_ext, ber

