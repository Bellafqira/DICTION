from copy import deepcopy

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

import util.util
from networks.customTensorDataset import CustomTensorDataset
from networks.linear_mod import DeepSigns
from util.gmm import GaussianMixture
from util.metric import Metric
from util.util import TrainModel, Random
import random
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gmm_compute(x_fc, n_features, n_components, path):
    """
    get the activation maps of all data and define a GaussianMixture to model the distribution of the activation maps
    :param x_fc: activation maps
    :param n_features: the size of each activation map
    :param n_components: number of classes
    :param path: the path where the GMM saved
    :return:
    """
    gmm = GaussianMixture(n_components=n_components, n_features=n_features)
    gmm.fit(x_fc)
    gmm_classes = gmm.predict(x_fc)
    hist = torch.histc(gmm_classes.float(), bins=n_components, min=0, max=n_components)
    h = hist.type(torch.int).numpy()

    gmm_dict = {"gmm": gmm.state_dict(), "x_fc": x_fc, "hist": h, "nonzero": np.nonzero(h)}
    torch.save(gmm_dict, path)


def gmm_load(n_features, n_components, path='idk'):
    gmm = GaussianMixture(n_components=n_components, n_features=n_features)
    gmm_dict = torch.load(path)
    gmm.load_state_dict(gmm_dict["gmm"])
    return gmm, gmm_dict["x_fc"], gmm_dict["hist"], gmm_dict["nonzero"]


def subset_training_data(watermarked_classes, x_train, y_gmm, y_train, percent):
    # what expect the Deepsigns paper
    nb_samplers = int(len(x_train) * percent)
    indices_gmm = torch.cat([torch.where(y_gmm == t)[0] for t in watermarked_classes])

    x_train_tmp = x_train[indices_gmm]
    y_train_tmp = y_train[indices_gmm]

    var_ind_train = [torch.where(y_train_tmp == t)[0] for t in watermarked_classes]
    for var in var_ind_train:
        if len(var) == 0:
            raise Exception("There is an empty GMM class")

    indices_gmm_train = torch.cat(var_ind_train)
    if len(indices_gmm_train) == 0:
        raise Exception("There is no samples with the same y_train and y_gmm for the selected means GMM")

    if nb_samplers < indices_gmm_train.shape[0]:
        ind = random.sample(range(len(indices_gmm_train)), nb_samplers)
        key_index = indices_gmm_train[ind]
        x_key, y_key = x_train_tmp[key_index], y_train_tmp[key_index]
        print("x_key size = ", len(key_index))
    else:
        x_key, y_key = x_train_tmp[indices_gmm_train], y_train_tmp[indices_gmm_train]
        print("x_key size = ", len(indices_gmm_train))

    return x_key, y_key


def get_trigger_set(gmm, x_fc, train_data, train_labels, watermarked_classes, percent):
    # Now let's get for each sample in X it's corresponding GMM class
    y_gmm = gmm.predict(x_fc)
    # Let's now select a subset of the training data for the  WM embedding.
    x_key, y_key = subset_training_data(watermarked_classes, train_data, y_gmm, train_labels, percent=percent)

    return x_key, y_key


# Computing loss1
def mu_loss1(x_fc, mu, mu_bar, watermarked_classes, y_key):
    index = torch.tensor([[i, j] for i in range(len(mu)) for j in range(len(mu_bar))])

    # increase the distance between carrier classes and non-carriers
    loss_ = Metric.mse(mu[index.T[0]], mu_bar[index.T[1]])  #

    act = torch.stack(
        [torch.mean(x_fc[torch.where(y_key == t)[0]], dim=0) for t in watermarked_classes])
    # approaching statistical means to the carrier GMM means
    gmm_loss = Metric.mse(act.cuda(), mu.cuda())
    return gmm_loss, loss_


def embed(train_loader, init_model, test_loader, x_key, y_key, gmm,
          watermarked_classes, config) -> object:
    # Generate the watermark
    watermark = 1. * torch.randint(0, 2, size=(config["nb_wat_classes"], config["watermark_size"]))
    # print("watermark", watermark)
    """generate matrix matrix_a"""
    matrix_a = Random.generate_secret_matrix(config["n_features"], config["watermark_size"])

    """"updating embed with config_embed"""
    model = deepcopy(init_model)
    nodes, _ = get_graph_node_names(init_model)
    # print(nodes)
    # extract the features of the given layer
    extractor = create_feature_extractor(model, [config["layer_name"]])
    criterion = config["criterion"]
    # DeepSigns model
    # first 1) Keep trace of mu_tbar to avoid their change at the time of the training
    mu = gmm.mu.squeeze(0)
    t_bar = [i for i in range(config["n_components"]) if i not in watermarked_classes]
    mu_bar = mu[t_bar]
    # initiate the deepSigns model with the means of GMM watermarked classes
    model_deepSigns = DeepSigns(mu[watermarked_classes])

    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': config["lr"], 'weight_decay': config["wd"]},
        {'params': model_deepSigns.parameters(), 'lr': 1e-1, 'weight_decay': 1e-2}
    ], lr=config["lr"])

    acc = TrainModel.evaluate(model, test_loader, config)
    # ber_ design the ber computed by the statistical means
    ber_ = 1

    x_key_, y_key_ = 0, 0
    loss, gmm_loss, mu_loss = 1, 1, 1

    matrix_g = model_deepSigns(matrix_a)

    while Metric.bce_(matrix_g, watermark).item() > 100:
        matrix_a = torch.randn_like(matrix_a)
        matrix_g = model_deepSigns(matrix_a)

    transform_train = transforms.Compose([
        # transforms.RandomCrop(size=32, padding=4),
        # transforms.RandomHorizontalFlip(),
        # AddGaussianNoise(config["mean"], config["std"]),
    ])
    dataset_key = CustomTensorDataset(x_key, y_key, transform_list=transform_train)
    key_loader = DataLoader(dataset=dataset_key, batch_size=len(x_key), shuffle=True)

    for epoch in range(config["epochs"]):
        train_loss = correct = total = 0
        loop = tqdm(train_loader, leave=True)

        for batch_idx, (x_train, y_train) in enumerate(loop):
            # Get activation maps
            x_key_, y_key_ = next(iter(key_loader))
            x_fc = extractor(x_key_)[config["layer_name"]]
            # Get the GMM means without changing the non-carriers GMM means.
            mu_dp = next(model_deepSigns.parameters())
            # Get train data
            x_train, y_train = x_train.to(config["device"]), y_train.to(config["device"])
            # x_train = torch.cat((x_train, x_key_))
            # y_train = torch.cat((y_train, y_key_.cuda()))
            y_train = y_train.type(torch.long)

            optimizer.zero_grad()

            # for each gmm class i in T the first term of loss1 here measures the distance between mu[i]
            # and the statistical mean of X_key sample corresponding to class i
            gmm_loss, mu_loss = mu_loss1(x_fc, mu_dp, mu_bar, watermarked_classes, y_key_)
            # sanity check
            assert gmm_loss.requires_grad == True, 'broken computational graph :/'
            matrix_g = model_deepSigns(matrix_a)

            assert Metric.bce_(matrix_g, watermark).requires_grad == True, 'broken computational graph :/'
            # λ1, λ2 control the trade of between WM embedding and the accuracy of the model
            y_pred = model(x_train)
            loss0 = criterion(y_pred, y_train)
            loss1 = gmm_loss - 0.0001 * mu_loss
            loss2 = Metric.bce_(matrix_g, watermark)
            loss = loss0 + config["lambda_1"] * loss1 + config["lambda_2"] * loss2

            train_loss += loss0.item()
            _, predicted = y_pred.max(1)
            total += y_train.size(0)
            correct += predicted.eq(y_train).sum().item()

            assert loss.requires_grad == True, 'broken computational graph :/'
            loss.backward(retain_graph=True)
            optimizer.step()

            ber = _get_ber(matrix_g, watermark)
            _, ber_ = _extract(x_fc, y_key_, watermarked_classes, matrix_a, watermark, print_ber=False)

            loop.set_description(f"Epoch [{epoch}/{config['epochs']}]")
            loop.set_postfix(loss=train_loss / (batch_idx + 1), acc=100. * correct / total,
                             correct_total=f"[{correct}"
                                           f"/{total}]", loss2=f"{loss2:1.4f}", loss1=f"{loss1:1.4f}",
                             ber=f"{ber:1.3f}",
                             ber_=f"{ber_:1.3f}")
        if epoch % 10 == 0:
            with torch.no_grad():
                ber = _get_ber(matrix_g, watermark)
                x_fc = stack_x_fc(extractor, x_key_, config["batch_size"], config)
                _, ber_ = _extract(x_fc, y_key_, watermarked_classes, matrix_a, watermark, print_ber=False)

                acc = TrainModel.evaluate(model, test_loader, config)
                print(
                    f"epoch:{epoch}---loss: {loss.item():1.3f}---ber: {ber:1.3f}---ber_mean: {ber_:1.3f}---gmm_loss: "
                    f"{gmm_loss:1.4f}---mu_sep: {mu_loss:.3f}---acc: {acc}")

        if ber_ == 0 and epoch >= config["epoch_check"]:
            print("saving!")
            supplementary = {'model': model, 'key_matrix': matrix_a, 'watermark': watermark,
                             'watermarked_classes': watermarked_classes,
                             'key_loader': key_loader, 'ber': ber,
                             "layer_name": config["layer_name"]}
            TrainModel.save_model(deepcopy(model), acc, epoch, config['save_path'], supplementary)
            break

    return model, ber_


def extract(model_watermarked, supp):
    extractor = create_feature_extractor(model_watermarked, [supp["layer_name"]])
    x_key, y_key = next(iter(supp["key_loader"]))
    x_fc = extractor(x_key)[supp["layer_name"]]
    return _extract(x_fc, y_key, supp["watermarked_classes"], supp["key_matrix"], supp["watermark"],
                    print_ber=True)


def _extract(x_fc, y_k, watermarked_classes, matrix_a, watermark, print_ber=True):
    """ This function allows the detection"""
    mu_ext = torch.stack([torch.mean(x_fc[torch.where(y_k == t)[0]], dim=0) for t in watermarked_classes])
    g_ext = torch.nn.Sigmoid()(mu_ext @ matrix_a.cuda()).cpu()
    b_ext = (g_ext > 0.5) * 1.
    ber = Metric.get_ber(b_ext, watermark)
    if print_ber:
        print(f'BER after extraction = {ber}')
    return b_ext, ber


def stack_x_fc(extractor, x_key, batch_size, config):
    x_fc = torch.cat([extractor(x_key[i: i + batch_size].to(config["device"]))[config["layer_name"]] for i in
                      range(0, len(x_key), batch_size)], dim=0)
    return x_fc


def _get_ber(matrix_g, watermark):
    b_ext = (matrix_g > 0.5) * 1.
    ber = abs(b_ext - watermark).mean()
    return ber
