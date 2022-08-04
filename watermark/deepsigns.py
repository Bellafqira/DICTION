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
from util.util import TrainModel
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
    gmm.mu = torch.nn.Parameter(gmm.mu, requires_grad=True)
    return gmm, gmm_dict["x_fc"], gmm_dict["hist"], gmm_dict["nonzero"]


# def subset_training_data(watermarked_classes, x_train, y_gmm, y_train, percent):

#     nb_samplers = int(len(x_train) * percent)

#     indices = torch.cat([torch.where(y_gmm == t)[0] for t in watermarked_classes])

#     ind = random.sample(range(indices.shape[0]), nb_samplers)

#     key_index = indices[ind]

#     x_key, y_key = x_train[key_index], y_train[key_index]

#     return key_index, x_key, y_key


# def subset_training_data(watermarked_classes, x_train, y_gmm, y_train, percent):
#     nb_samplers = int(len(x_train) * percent)
#
#     indices = torch.cat([torch.where(y_gmm == t)[0] for t in watermarked_classes])
#
#     ind = random.sample(range(indices.shape[0]), nb_samplers)
#
#     key_index = indices[ind]
#
#     x_key, y_key = x_train[key_index], y_train[key_index]
#
#     return key_index, x_key, y_key


def subset_training_data(watermarked_classes, x_train, y_gmm, y_train, percent):
    # what expect the Deepsigns paper
    nb_samplers = int(len(x_train) * percent)
    indices_gmm = torch.cat([torch.where(y_gmm == t)[0] for t in watermarked_classes])

    x_train_tmp = x_train[indices_gmm]
    y_train_tmp = y_train[indices_gmm]

    indices_gmm_train = torch.cat([torch.where(y_train_tmp == t)[0] for t in watermarked_classes])

    if nb_samplers < indices_gmm_train.shape[0]:
        ind = random.sample(range(indices_gmm_train.shape[0]), nb_samplers)
        key_index = indices_gmm_train[ind]
        x_key, y_key = x_train_tmp[key_index], y_train_tmp[key_index]
        print("here1", len(key_index))
    else:
        key_index = indices_gmm_train
        x_key, y_key = x_train_tmp[indices_gmm_train], y_train_tmp[indices_gmm_train]
        print("here2", len(indices_gmm_train))
        if len(indices_gmm_train) == 0:
            raise Exception("There is no samples with the same y_train and y_gmm for the selected means GMM")
    return key_index, x_key, y_key


# def subset_training_data_update(watermarked_classes, x_train, y_gmm, y_train, percent):
#     nb_samples = int(len(x_train) * percent)
#     nb_samples_t = nb_samples // len(watermarked_classes)
#     indices = [torch.where(y_gmm == t)[0] for t in watermarked_classes]
#     key_index = torch.tensor([])
#     for ind in indices:
#         pos = random.choices(range(ind.shape[0]), k=nb_samples_t)
#         key_index = torch.cat((key_index, ind[pos]))
#     key_index = key_index.type(torch.int64)
#     x_key, y_key = x_train[key_index], y_train[key_index]
#     return key_index, x_key, y_key

#
# def subset_training_data_update(watermarked_classes, x_train, y_gmm, y_train, percent):
#     nb_samples = int(len(x_train) * percent)
#     nb_samples_t = nb_samples // len(watermarked_classes)
#
#     indices_ = (y_gmm == y_train).float().nonzero().flatten()
#     y_inter = y_gmm[indices_]
#     y_ii = y_train[indices_]
#     print("hereee", y_ii)
#     indices = [torch.where(y_inter == t)[0] for t in watermarked_classes]
#
#     key_index = torch.tensor([])
#     for ind in indices:
#         pos = random.choices(range(ind.shape[0]), k=nb_samples_t)
#         key_index = torch.cat((key_index, ind[pos]))
#     key_index = key_index.type(torch.int64)
#     x_key, y_key = x_train[key_index], y_train[key_index]
#     return key_index, x_key, y_key


def get_trigger_set(gmm, x_fc, train_data, train_labels, watermarked_classes, percent):
    # Now let's get for each sample in X it's corresponding GMM class
    y_gmm = gmm.predict(x_fc)
    # Let's now select a subset of the training data for the  WM embedding.
    key_index, x_key, y_key = subset_training_data(watermarked_classes, train_data, y_gmm, train_labels,
                                                   percent=percent)
    # gmm.mu = torch.nn.Parameter(gmm.mu, requires_grad=True)

    # y_k are the subset of gmm classes corresponding to x_key
    y_k = y_gmm[key_index]

    return y_k, key_index, x_key, y_key


# Computing loss1
def mu_loss1(x_fc1, mu, mu_bar, watermarked_classes, y_k, nb_components=10):  # y_key
    # mu_t = mu[watermarked_classes].cuda()
    mu_t = mu
    t_bar = [t for t in range(nb_components) if t not in watermarked_classes]
    index = torch.tensor([[i, j] for i in range(len(watermarked_classes)) for j in range(len(t_bar))])

    # increase the distance between carrier classes and non-carriers
    loss_ = Metric.mse(mu[index.T[0]], mu_bar[index.T[1]].detach())  #
    # print("here3 saas, ", [Metric.mse(mu[[t]*x_fc1[y_k == t].shape[0]].cuda(), x_fc1[y_k == t].cuda())
    #                        for t in [8, 0, 3, 9, 7, 2, 5, 1]])

    # gmm_loss = torch.stack([Metric.mse(x_fc1[y_k == t].cuda(), mu[[t]*x_fc1[y_k == t].shape[0]].cuda())
    #                         for t in watermarked_classes.tolist()]).mean()

    act = torch.stack(
        [torch.mean(x_fc1[torch.where(y_k == t)[0]], dim=0) for t in watermarked_classes])
    # approaching statistical means to the carrier GMM means
    gmm_loss = Metric.mse(act.cuda(), mu_t.cuda())
    return gmm_loss, loss_


def embed(train_loader, train_labels, init_model, test_loader, y_k, x_key, y_key, gmm, matrix_a, watermark,
          watermarked_classes, config) -> object:
    """"updating embed with config_embed"""
    model = deepcopy(init_model)
    nodes, _ = get_graph_node_names(init_model)
    print(nodes)
    # extract the features of the given layer
    extractor = create_feature_extractor(model, [config["layer_name"]])
    criterion = config["criterion"]
    # optimizer = util.util.TrainModel.get_optimizer(model, config, gmm.parameters())
    # DeepSigns model
    # first 1) Keep trace of mu_tbar to avoid their change at the time of the training
    t_bar = [i for i in range(config["n_components"]) if i not in watermarked_classes]
    mu_bar = gmm.mu.squeeze(0)[t_bar]
    # initiate the deepSigns model with the means of GMM watermarked classes
    mu = gmm.mu.squeeze(0)
    model_deepSigns = DeepSigns(mu[watermarked_classes])

    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': config["lr"], 'weight_decay': config["wd"]},
        {'params': model_deepSigns.parameters(), 'lr': 1e-1, 'weight_decay': 1e-4}
    ], lr=config["lr"])

    acc = TrainModel.evaluate(model, test_loader, config)
    # ber_ design the ber computed by the statistical means
    ber_ = 1

    # matrix_g = torch.nn.Sigmoid()(mu1[watermarked_classes] @ matrix_a)
    matrix_g = model_deepSigns(matrix_a)
    # 1, 2, 5, 20 for wat <=16, =32, =64, =128
    while Metric.bce_(matrix_g, watermark[[0] * config["nb_wat_classes"]]).item() > 100:
        matrix_a = torch.randn_like(matrix_a)
        matrix_g = model_deepSigns(matrix_a)
    # start training
    # read secret data
    # x_train, y_train = x_key.to(config["device"]), y_key.to(config["device"])
    # y_train = y_train.type(torch.long)

    # batch_size = 512
    # train_loader = [
    #     (x_key[i:i + batch_size].to(config["device"]), x_key[i:i + batch_size].to(config["device"]))
    #     for i in range(0, x_key.shape[0], batch_size)]

    transform_train = transforms.Compose([
        # transforms.RandomCrop(size=32, padding=4),
        # transforms.RandomHorizontalFlip(),
        # AddGaussianNoise(config["mean"], config["std"]),
    ])
    dataset_key = CustomTensorDataset(x_key, y_key, transform_list=transform_train)
    key_loader = DataLoader(dataset=dataset_key, batch_size=config["batch_size"], shuffle=True)

    for epoch in range(config["epochs"]):
        # read secret data
        train_loss = correct = total = 0
        loop = tqdm(train_loader, leave=True)

        for batch_idx, (x_train, y_train) in enumerate(loop):
            # Get activation maps
            x_key_, y_key_ = next(iter(key_loader))
            x_fc = extractor(x_key_)[config["layer_name"]]
            # Get the GMM means without changing the non-carriers GMM means.
            mu = next(model_deepSigns.parameters())
            # Get train data
            x_train, y_train = x_train.to(config["device"]), y_train.to(config["device"])
            x_train = torch.cat((x_train, x_key_))
            y_train = torch.cat((y_train, y_key_.cuda()))
            y_train = y_train.type(torch.long)

            optimizer.zero_grad()

            # for each gmm class i in T the first term of loss1 here measures the distance between mu[i]
            # and the statistical mean of X_key sample corresponding to class i
            # gmm_loss, mu_loss = mu_loss1(x_fc, mu1, watermarked_classes, y_key, nb_components=config["n_components"]) #deepsigns
            gmm_loss, mu_loss = mu_loss1(x_fc, mu, mu_bar, watermarked_classes, y_key_,
                                         nb_components=config["n_components"])
            # modify y_k
            # sanity check
            assert gmm_loss.requires_grad == True, 'broken computational graph :/'

            # matrix_g = torch.nn.Sigmoid()(mu1[watermarked_classes] @ matrix_a)
            matrix_g = model_deepSigns(matrix_a)
            assert Metric.bce_(matrix_g, watermark).requires_grad == True, 'broken computational graph :/'
            # λ1, λ2 control the trade of between WM embedding and the accuracy of the model
            y_pred = model(x_train)
            loss0 = criterion(y_pred, y_train)
            loss1 = gmm_loss - 0.001 * mu_loss
            loss2 = Metric.bce_(matrix_g, watermark)
            loss = loss0 + config["lambda_1"] * loss1 + config["lambda_2"] * loss2

            # ber = _get_ber(matrix_g, watermark)
            # print(Metric.bce_(matrix_g, watermark[[0] * config["nb_wat_classes"]]).item())
            # print(watermarked_classes)
            assert loss.requires_grad == True, 'broken computational graph :/'

            loss.backward(retain_graph=True)
            optimizer.step()

            train_loss += loss0.item()
            _, predicted = y_pred.max(1)
            total += y_train.size(0)
            correct += predicted.eq(y_train).sum().item()

            ber = _get_ber(matrix_g, watermark)
            _, ber_ = _extract(x_fc, y_key_, watermarked_classes, matrix_a, watermark, print_ber=False)

            loop.set_description(f"Epoch [{epoch}/{config['epochs']}]")
            loop.set_postfix(loss=train_loss / (batch_idx + 1), acc=100. * correct / total,
                             correct_total=f"[{correct}"
                                           f"/{total}]", loss2=f"{loss2:1.4f}", loss1=f"{loss1:1.4f}",
                             ber=f"{ber:1.3f}",
                             ber_=f"{ber_:1.3f}")
        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                ber = _get_ber(matrix_g, watermark)
                # x_fc = stack_x_fc(extractor, x_key, 100, config)
                # _, ber_ = _extract(x_fc, y_key, watermarked_classes, matrix_a, watermark, print_ber=False)

                x_fc = stack_x_fc(extractor, x_key_, config["batch_size"], config)
                _, ber_ = _extract(x_fc, y_key_, watermarked_classes, matrix_a, watermark, print_ber=False)

                acc = TrainModel.evaluate(model, test_loader, config)
                print(
                    f"epoch:{epoch}---loss: {loss.item():1.3f}---ber: {ber:1.3f}---ber_mean: {ber_:1.3f}---gmm_loss: "
                    f"{gmm_loss:1.4f}---mu_sep: {mu_loss:.3f}---acc: {acc}")

        if ber_ == 0 and epoch % 10 == 0:
            print("saving!")
            supplementary = {'model': model, 'key_matrix': matrix_a, 'watermark': watermark,
                             'watermarked_classes': watermarked_classes,
                             'x_key': x_key_, 'y_key': y_key_, 'y_k': y_k, 'ber': ber,
                             "layer_name": config["layer_name"]}
            TrainModel.save_model(deepcopy(model), acc, epoch, config['save_path'], supplementary)
            break

    return model, ber_


def extract(model_watermarked, supp):
    extractor = create_feature_extractor(model_watermarked, [supp["layer_name"]])
    x_fc = extractor(supp["x_key"])[supp["layer_name"]]
    return _extract(x_fc, supp["y_key"], supp["watermarked_classes"], supp["key_matrix"], supp["watermark"],
                    print_ber=True)


def _extract(x_fc1, y_k, watermarked_classes, matrix_a, watermark, print_ber=True):
    """ This function allows the detection"""
    μ_extracted = torch.stack([torch.mean(x_fc1[torch.where(y_k == t)[0]], dim=0) for t in watermarked_classes])
    g_extracted = torch.nn.Sigmoid()(μ_extracted @ matrix_a.cuda()).cpu()
    b_extracted = (g_extracted > 0.5) * 1.
    ber = Metric.get_ber(b_extracted, watermark)
    if print_ber:
        print(f'BER with fp = {ber}')
    return b_extracted, ber * 100


def stack_x_fc(extractor, x_key, batch_size, config):
    x_fc = torch.cat([extractor(x_key[i: i + batch_size].to(config["device"]))[config["layer_name"]] for i in
                      range(0, x_key.shape[0],
                            batch_size)], dim=0)
    return x_fc


def _get_ber(matrix_g, watermark):
    b_ext = (matrix_g > 0.5) * 1.
    # b_ext = (b_ext.mean(axis=0) > 0.5) * 1
    ber = abs(b_ext - watermark).mean()
    return ber
