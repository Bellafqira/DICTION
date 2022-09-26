from copy import deepcopy
import random
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from tqdm import tqdm
from networks.customTensorDataset import CustomTensorDataset
from networks.linear_mod import LinearMod
from util.metric import Metric

# Check Device configuration
from util.util import TrainModel, Random


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def embed(init_model, test_loader, train_loader, config) -> object:
    # Generate a random watermark to insert
    watermark = Random.get_rand_bits(config["watermark_size"], 0., 1.)
    watermark = torch.tensor(watermark).reshape(1, config["watermark_size"])

    # Instance the linear mod
    linear_mod = LinearMod(config).to(config["device"])
    # Instance the model
    model = deepcopy(init_model)
    # The parameters of training
    criterion = config["criterion"]
    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': config["lr"], 'weight_decay': config["wd"]},
        {'params': linear_mod.parameters(), 'lr': 1e-2, 'weight_decay': 1e-4}
    ], lr=config["lr"])
    scheduler = TrainModel.get_scheduler(optimizer, config)
    # Get the activation layer of the original model and make sure that its parameters are not trainable
    init_extractor = create_feature_extractor(init_model, [config["layer_name"]])
    for param in init_extractor.parameters():
        param.requires_grad = False
    # Proj model
    # print(get_graph_node_names(model))
    extractor = create_feature_extractor(model, [config["layer_name"]])
    # Select the indices of the activations maps that will be fed to the projection model
    indices = random.choices(range(config["n_features_layer"]), k=config["n_features"])
    # Latent space
    x_key, y_key = next(iter(train_loader))
    x_key = torch.normal(mean=config["mean"], std=config["std"], size=x_key.shape)

    transform_train = transforms.Compose([
        # transforms.RandomCrop(size=32, padding=4),
        # transforms.RandomHorizontalFlip(),
        AddGaussianNoise(config["mean"], config["std"]),
    ])
    dataset_key = CustomTensorDataset(x_key, y_key, transform_list=transform_train)
    key_loader = DataLoader(dataset=dataset_key, batch_size=config["batch_size"], shuffle=True)

    # rd_watermark = Random.get_rand_bits(config["watermark_size"], 1., 0.)
    # rd_watermark = torch.tensor(rd_watermark).reshape(1, config["watermark_size"])

    ber_ = 1
    ber = 1
    loss = 1
    loss_2 = 1
    for epoch in range(config["epochs"]):
        embed_loss = train_loss = correct = total = 0

        loop = tqdm(train_loader, leave=True)
        rd_watermark = Random.get_rand_bits(config["watermark_size"], 1., 0.)
        rd_watermark = torch.tensor(rd_watermark).reshape(1, config["watermark_size"])
        for batch_idx, (x_train, y_train) in enumerate(loop):
            y_train = y_train.type(torch.long)
            x_train, y_train = x_train.to(config["device"]), y_train.to(config["device"])
            optimizer.zero_grad()
            y_pred = model(x_train)

            x_key, _ = next(iter(key_loader))
            x_key = x_key.cuda()

            # Get activation maps of the watermarked model
            x_fc = extractor(x_key)[config["layer_name"]]
            act = torch.index_select(x_fc, 1, torch.tensor(indices).cuda())
            act = torch.stack([torch.mean(act, dim=0)])
            # act = nn.functional.normalize(act).cpu()

            # Get activation maps of the original model
            init_x_fc = init_extractor(x_key)[config["layer_name"]]
            init_act = torch.index_select(init_x_fc, 1, torch.tensor(indices).cuda())
            init_act = torch.stack([torch.mean(init_act, dim=0)])
            # init_act = nn.functional.normalize(init_act).cpu()

            out_watermark = linear_mod(act)
            init_out_watermark = linear_mod(init_act)

            loss_1 = criterion(y_pred, y_train)
            loss_2_1 = Metric.bce_(out_watermark, watermark.cuda())
            loss_2_2 = Metric.bce_(init_out_watermark, rd_watermark.cuda())
            loss_2 = loss_2_1 + loss_2_2

            loss = loss_1 + config["lambda"] * loss_2
            assert loss.requires_grad == True, 'broken computational graph :/'

            loss.backward(retain_graph=True)
            optimizer.step()

            train_loss += loss_1.item()
            embed_loss += loss_2.item()
            _, predicted = y_pred.max(1)
            total += y_train.size(0)
            correct += predicted.eq(y_train).sum().item()

            # update the progress bar
            _, ber = _get_ber(out_watermark.cpu().detach().numpy(), watermark.cpu())
            _, ber_ = _extract(act, linear_mod, watermark)  # linear_mod.fc.bias
            loop.set_description(f"Epoch [{epoch}/{config['epochs']}]")
            loop.set_postfix(loss=train_loss / (batch_idx + 1), acc=100. * correct / total,
                             correct_total=f"[{correct}"
                                           f"/{total}]", loss_2=f"{embed_loss / (batch_idx + 1):1.4f}",
                             ber=f"{ber:1.3f}",
                             ber_=f"{ber_:1.3f}", loss21=f"{loss_2_1:1.3f}", loss22=f"{loss_2_2:1.3f}")
        scheduler.step()

        if (epoch + 1) % config["epoch_check"] == 0:
            with torch.no_grad():
                # x_fc = stack_x_fc(extractor, x_key, config)
                x_key, _ = next(iter(key_loader))
                x_fc = extractor(x_key.cuda())[config["layer_name"]]
                act = torch.index_select(x_fc, 1, torch.tensor(indices).cuda())
                act = torch.stack([torch.mean(act, dim=0)])
                # act = nn.functional.normalize(act).cpu()

                _, ber_ = _extract(act, linear_mod, watermark.cuda())  # linear_mod.fc.bias
                acc = TrainModel.evaluate(model, test_loader, config)
                print(
                    f"epoch:{epoch}---loss: {loss.item():1.7f}---ber_mean: {ber_:1.3f}"
                    f"---param_var_loss: "
                    f"{loss_2:1.4f}---acc: {acc}")

            if ber_ == 0:
                print("saving... watermarked model ")
                supplementary = {'model': model, 'key_matrix': linear_mod, 'watermark': watermark,
                                 'x_key': key_loader, 'y_key': y_key, 'ber': ber,
                                 "layer_name": config["layer_name"], "indices": indices}
                TrainModel.save_model(deepcopy(model), acc, epoch, config['save_path'], supplementary)
                break

    return model, ber_


def extract(model_watermarked, supp):
    model_watermarked.eval()
    extractor = create_feature_extractor(model_watermarked, [supp["layer_name"]])
    ber_total = torch.zeros(size=supp["watermark"].shape)
    b_ext = 1
    for x_key, _ in supp["x_key"]:
        x_fc = extractor(x_key.cuda())[supp["layer_name"]]
        act = torch.index_select(x_fc.cpu(), 1, torch.tensor(supp["indices"]))
        act = torch.stack([torch.mean(act, dim=0)])
        # act = nn.functional.normalize(act).cpu()
        b_ext, ber = _extract(act.cuda(), supp["key_matrix"], supp["watermark"])
        ber_total += b_ext
    ber_total = 1. * (ber_total > 0.5)
    ber = Metric.get_ber(ber_total, supp["watermark"])
    return b_ext, ber


def _extract(act, model, watermark):
    """ This function allows the detection"""
    watermark_out = model(act)
    # watermark_out = torch.stack([torch.mean(watermark_out, dim=0)])
    b_ext, ber = _get_ber(watermark_out.cpu().detach().numpy(), watermark.cpu())

    return b_ext, ber


def _get_ber(wat_out, watermark):
    b_ext = 1. * (wat_out > 0.5)
    ber = Metric.get_ber(b_ext, watermark)
    return b_ext, ber


def stack_x_fc(extractor, x_key, config):
    x_fc = torch.cat([extractor(x_key[i: i + config["batch_size"]].to(config["device"]))[config["layer_name"]].detach()
                     .cpu() for i in range(0, x_key.shape[0], config["batch_size"])], dim=0)
    return x_fc.cuda()

