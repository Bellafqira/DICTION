import time
from copy import deepcopy

import numpy as np
import torch
from torch import optim
from torch.nn import BCELoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

from networks.linear_mod import DeepSigns
from util.gmm import GaussianMixture
from util.metric import Metric
from util.util import TrainModel, Random, CustomTensorDataset, Util
import random
from tqdm import tqdm


# Computing loss1
def mu_loss1(act, mu, mu_bar, watermarked_classes, y_key):

    index = torch.tensor([[i, j] for i in range(len(mu)) for j in range(len(mu_bar))])
    # increase the distance between carrier classes and non-carriers
    loss_ = torch.nn.MSELoss(reduction='mean')(mu[index.T[0]], mu_bar[index.T[1]])  #

    act = torch.stack(
        [torch.mean(act[torch.where(y_key == t)[0]], dim=0) for t in watermarked_classes])
    # approaching statistical means to the carrier GMM means
    gmm_loss = torch.nn.MSELoss(reduction='sum')(act.cuda(), mu.cuda())
    return gmm_loss, loss_

def get_trigger_set(gmm, act, x_train, watermarked_classes, percent):
    # Now let's get for each sample in X it's corresponding GMM class
    y_gmm = gmm.predict(act)
    # Let's now select a subset of the training data for the  WM embedding.
    # what expect the Deepsigns paper
    nb_samplers = int(len(x_train) * percent)
    indices_gmm = torch.cat([torch.where(y_gmm == t)[0] for t in watermarked_classes])

    x_train_tmp = x_train[indices_gmm]
    y_train_tmp = y_gmm[indices_gmm]

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


def embed(init_model, test_loader, train_loader, config) -> object:
    # let's start with the GMM computation
    # define the activation maps extractor :
    extractor = create_feature_extractor(init_model, [config["layer_name"]])
    # Get data, labels and activation maps
    train_data, train_labels, act = [], [], []
    for images, label in train_loader:
        images = images.to(config["device"])
        train_data.append(images)
        train_labels.append(label)
        act.append(extractor(images)[config["layer_name"]].detach().cpu())
    train_data = torch.cat(train_data).cuda()
    train_labels = torch.cat(train_labels)
    act = torch.cat(act).cuda()

    config["n_features"] = len(act[0])

    print("train_data.shape: ", train_data.shape, "train_labels.shape: ", train_labels.shape, "act.shape: ", act.shape)
    # Now get the gmm to generate the trigger set
    gmm, _, _, gmm_nonzero = _gmm(act, config)
    # T in the paper
    watermarked_classes = random.sample([i for i in gmm_nonzero[0]], config["nb_wat_classes"])
    watermarked_classes = torch.tensor(watermarked_classes)
    print("watermarked_classes ", watermarked_classes)
    # generate the trigger set
    x_key, y_key = get_trigger_set(gmm=gmm, act=act, x_train=train_data,
                                   watermarked_classes=watermarked_classes,
                                   percent=config["percent_ts"])

    # Batch_size takes all samples in x_key
    batch_size = len(x_key)
    # Generate the watermark
    watermark = 1. * torch.randint(0, 2, size=(config["nb_wat_classes"], config["watermark_size"]))

    # generate matrix matrix_a
    matrix_a = 1. * torch.randn(config["n_features"], config["watermark_size"], requires_grad=False).cuda()

    # updating embed with config_embed
    model = deepcopy(init_model)
    nodes, _ = get_graph_node_names(init_model)

    # extract the features of the given layer
    extractor = create_feature_extractor(model, [config["layer_name"]])
    criterion = config["criterion"]

    # DeepSigns model
    # first 1) Keep trace of mu_tbar to avoid their change at the time of the training
    mu = gmm.mu.squeeze(0)
    t_bar = [i for i in range(config["n_components"]) if i not in watermarked_classes]
    mu_bar = mu[t_bar]
    mu_bar.requires_grad = False
    # initiate the deepSigns model with the means of GMM watermarked classes
    model_deepSigns = DeepSigns(mu[watermarked_classes])

    optimizer = optim.Adam([
        {'params': model.parameters(), 'lr': config["lr"], 'weight_decay': config["wd"]},
        {'params': model_deepSigns.parameters(), 'lr': config["lr_DS"], 'weight_decay': config["wd_DS"]}
    ], lr=config["lr"])

    # ber_ design the ber computed by the statistical means
    ber_ = 1
    loss, gmm_loss, mu_loss = 1, 1, 1

    matrix_g = model_deepSigns(matrix_a)
    transform_train = transforms.Compose([
    ])
    dataset_key = CustomTensorDataset(x_key, y_key, transform_list=transform_train)
    key_loader = DataLoader(dataset=dataset_key, batch_size=batch_size, shuffle=False)
    # Get activation maps
    x_key_, y_key_ = next(iter(key_loader))

    # Schedulers
    model_scheduler = TrainModel.get_scheduler(optimizer, config)

    for epoch in range(config["epochs"]):
        train_loss = correct = total = 0
        loop = tqdm(train_loader, leave=True)
        model_deepSigns.train(True)
        model.train(True)

        for batch_idx, (x_train, y_train) in enumerate(loop):
            act = extractor(x_key_)[config["layer_name"]]
            # Get the GMM means without changing the non-carriers GMM means.
            mu_dp = model_deepSigns.var_param

            # Get train data
            x_train, y_train = x_train.to(config["device"]), y_train.to(config["device"])
            y_train = y_train.type(torch.long)

            optimizer.zero_grad()
            # for each gmm class i in T the first term of l_mu_act here measures the distance between mu[i]
            # and the statistical mean of X_key sample corresponding to class i
            gmm_loss, mu_loss = mu_loss1(act, mu_dp, mu_bar, watermarked_classes, y_key_)
            # sanity check
            assert gmm_loss.requires_grad == True, 'broken computational graph :/'
            matrix_g = model_deepSigns(matrix_a)

            assert BCELoss(reduction='sum')(matrix_g, watermark.cuda()).requires_grad == True, 'broken computational graph :/'
            # λ1, λ2 control the trade of between WM embedding and the accuracy of the model
            y_pred = model(x_train)

            l_main_task = criterion(y_pred, y_train)
            l_mu_act = gmm_loss - mu_loss
            l_wat = BCELoss(reduction='sum')(matrix_g, watermark.cuda())
            loss = l_main_task + config["lambda_1"] * l_mu_act + config["lambda_2"] * l_wat

            train_loss += l_main_task.item()
            _, predicted = y_pred.max(1)
            total += y_train.size(0)
            correct += predicted.eq(y_train).sum().item()

            assert loss.requires_grad == True, 'broken computational graph :/'
            loss.backward(retain_graph=True)
            optimizer.step()

            ber = _get_ber(matrix_g, watermark.cuda())
            _, ber_ = _extract(act, y_key_, watermarked_classes, matrix_a, watermark)

            loop.set_description(f"Epoch [{epoch}/{config['epochs']}]")
            loop.set_postfix(l_main_task=train_loss / (batch_idx + 1), acc=100. * correct / total,
                             correct_total=f"[{correct}"
                                           f"/{total}]", l_wat=f"{l_wat:1.4f}", l_mu_act=f"{l_mu_act:1.4f}",
                             ber=f"{ber:1.3f}",
                             ber_=f"{ber_:1.3f}")

        model_scheduler.step()

        if (epoch + 1) % config["epoch_check"] == 0:
            with torch.no_grad():
                ber = _get_ber(matrix_g, watermark.cuda())
                act = extractor(x_key_)[config["layer_name"]]
                _, ber_ = _extract(act, y_key_, watermarked_classes, matrix_a, watermark)

                acc = TrainModel.evaluate(model, test_loader, config)
                print(
                    f"epoch:{epoch}---loss: {loss.item():1.3f}---ber: {ber:1.3f}---ber_mean: {ber_:1.3f}---gmm_loss: "
                    f"{gmm_loss:1.4f}---mu_sep: {mu_loss:.3f}---acc: {acc}")

            if ber_ == 0:
                print("saving... watermarked model ")
                supplementary = {'model': model, 'matrix_a': matrix_a, 'watermark': watermark,
                                 'watermarked_classes': watermarked_classes,
                                 'key_loader': key_loader, 'ber': ber,
                                 "layer_name": config["layer_name"]}
                TrainModel.save_model(deepcopy(model), acc, epoch, config['save_path'], supplementary)
                print("model saved!")
                break

    return model, ber_


def extract(model_watermarked, supp):
    extractor = create_feature_extractor(model_watermarked, [supp["layer_name"]])
    x_key, y_key = next(iter(supp["key_loader"]))
    act = extractor(x_key)[supp["layer_name"]]
    return _extract(act, y_key, supp["watermarked_classes"], supp["matrix_a"], supp["watermark"])


def _extract(act, y_key, watermarked_classes, matrix_a, watermark):
    mu_ext = torch.stack([torch.mean(act[torch.where(y_key == t)[0]], dim=0) for t in watermarked_classes])
    g_ext = torch.nn.Sigmoid()(mu_ext @ matrix_a.cuda()).cpu()
    b_ext = (g_ext > 0.5) * 1.
    ber = Metric.get_ber(b_ext, watermark)
    return b_ext, ber


def _get_ber(matrix_g, watermark):
    b_ext = (matrix_g > 0.5) * 1.
    ber = abs(b_ext - watermark).mean()
    return ber


def _gmm(act, config):
    # #####  computing the gmm #####
    # act = Util.stack_act(extractor, train_loader, config)
    print("activations shape : ", act.shape)
    # """compute and save the gmm"""
    print("start gmm")
    start = time.time()
    gmm = GaussianMixture(n_components=config["n_components"], n_features=config["n_features"])
    gmm.fit(act)
    print("GMM computed in  ", time.time() - start, "second")
    print("start loading the GMM")
    gmm_classes = gmm.predict(act)

    hist_gmm_classes = torch.histc(gmm_classes.float(), bins=config["n_components"], min=0, max=config["n_components"])
    gmm_hist = hist_gmm_classes.type(torch.int).numpy()
    gmm_nonzero = np.nonzero(gmm_hist)

    print("show histogram of GMM classes")
    print(gmm_hist)
    print("show non zero classes")
    print(gmm_nonzero)
    return gmm, act, gmm_hist, np.nonzero(gmm_hist)
