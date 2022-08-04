import random

import numpy as np
import torch
import time

from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

from attacks.pruning import pruning, print_sparsity
from util.util import TrainModel, Random, Database


from watermark.deepSigns_C import gmm_compute, gmm_load, get_trigger_set, extract, embed


class Tests:

    @staticmethod
    def gmm(config_embed, config_data):
        """computing the gmm"""
        # Get model
        init_model = TrainModel.get_model(config_embed["architecture"], config_embed["device"])
        init_model.load_state_dict(TrainModel.load_model(config_embed["path_model"] + ".pth")[0])
        init_model.eval()
        # Get data
        """Load data """
        train_loader, test_loader, attack_loader, remind_train_loader, attack_loader_size = \
            Database.load_split_dataset_loaders(config_data)
        print("Evaluation of the model to watermark")
        TrainModel.evaluate(init_model, test_loader, config_embed)
        # here we save the activation maps of all data in  the tensor x_fc
        # we fit then the GMM to this tensor
        extractor = create_feature_extractor(init_model, [config_embed["layer_name"]])
        x_fc = torch.tensor([])
        for idx_batch, data in enumerate(train_loader):
            images, label = data
            images = images.to(config_embed["device"])
            tmp = torch.cat([extractor(img[None, :])[config_embed["layer_name"]].detach().cpu() for img in images])
            x_fc = torch.cat((x_fc, tmp), 0)
        x_fc = x_fc.cuda()

        """compute and save the gmm"""
        print("start gmm")
        start = time.time()
        gmm_compute(x_fc=x_fc, n_features=config_embed["n_features"], n_components=config_embed["n_components"],
                    path=config_embed["path_gmm"])
        print("GMM computed in  ", time.time() - start, "second")
        print("start loading the GMM")
        gmm, x_fc1, gmm_hist, gmm_nonzero = gmm_load(config_embed["n_features"], config_embed["n_components"], path=config_embed["path_gmm"])
        print("show histogram of GMM classes")
        print(gmm_hist)
        print("show non zero classes")
        print(gmm_nonzero)

    @staticmethod
    def embedding(config_embed, config_data, nb_run=0):
        """embedding """
        """gmm parameters"""
        # Get model
        init_model = TrainModel.get_model(config_embed["architecture"], config_embed["device"])
        init_model.load_state_dict(TrainModel.load_model(config_embed["path_model"] + ".pth")[0])
        # Get data
        train_loader, test_loader, attack_loader, remind_train_loader, attack_loader_size = \
            Database.load_split_dataset_loaders(config_data)
        """evaluate the model"""
        TrainModel.evaluate(init_model, test_loader, config_embed)
        # T in the paper
        gmm, x_fc, _, gmm_nonzero = gmm_load(config_embed["n_features"], config_embed["n_components"],
                                             path=config_embed["path_gmm"])
        """Get the model, data and the paths of gmm and the model"""
        watermarked_classes = random.sample([i for i in gmm_nonzero[0]], config_embed["nb_wat_classes"])
        watermarked_classes = torch.tensor(watermarked_classes)

        print("here", watermarked_classes)
        """changing the batch_size"""
        train_data = torch.tensor([]).cuda()
        train_labels = torch.tensor([])
        for idx_batch, data in enumerate(train_loader):
            images, label = data
            images = images.to(config_embed["device"])
            tmp_data = torch.cat([img[None, :] for img in images])
            train_data = torch.cat((train_data, tmp_data), 0)
            train_labels = torch.cat((train_labels, label), 0)

        """generate the trigger set"""
        y_k, key_index, x_key, y_key = get_trigger_set(gmm=gmm, train_data=train_data, train_labels=train_labels,
                                                       watermarked_classes=watermarked_classes, x_fc=x_fc,
                                                       percent=config_embed["percent_ts"])


        """generate the watermark watermark"""
        # watermark = Random.get_rand_bits(config_embed["watermark_size"], 0., 1.)
        # # watermark = torch.ones(config_embed["watermark_size"])
        # watermark = torch.tensor(watermark).reshape(1, config_embed["watermark_size"])
        watermark = 1.*torch.randint(0, 2, size=(config_embed["nb_wat_classes"], config_embed["watermark_size"]))
        print("watermark", watermark)
        """generate matrix matrix_a"""
        matrix_a = Random.generate_secret_matrix(config_embed["n_features"], config_embed["watermark_size"])
        """embed the watermark with deepsign_c"""
        model_wat, ber = embed(train_loader, train_labels, init_model, test_loader, y_k, x_key, y_key, gmm,
                               matrix_a, watermark,
                               watermarked_classes, config_embed)

        acc = TrainModel.evaluate(model_wat, test_loader, config_embed)
        return acc, ber

    @staticmethod
    def fine_tune_attack(config_embed, config_attack, config_data):
        # Get data
        train_loader, test_loader, attack_loader, remind_train_loader, attack_loader_size = \
            Database.load_split_dataset_loaders(config_data)
        # load the watermarked model
        watermarked_model, acc, epoch, supp = TrainModel.load_model(config_embed['save_path'] + ".pth", supp=True)
        model_wat = supp["model"]
        # fine tune the watermarked model
        print("First check")
        TrainModel.evaluate(model_wat, test_loader, config_attack)

        # for name, param in model_wat.named_parameters():
        #     if name in ["fc2.weight", "fc2.bias"]:
        #         continue
        #     else:
        #         param.requires_grad = False
        for i in range(1, 4):
          model_wat = TrainModel.fine_tune(model_wat, train_loader, test_loader, config_attack)
          # check the accuracy and BER
          print(f"BER after finetuning {i*50}")
          extract(model_wat, supp)
          TrainModel.evaluate(model_wat, test_loader, config_attack)

    @staticmethod
    def pruning_attack(config_embed, config_attack, config_data):
        # Get data
        train_loader, test_loader, attack_loader, remind_train_loader, attack_loader_size = \
            Database.load_split_dataset_loaders(config_data)
        # load the watermarked model
        watermarked_model, acc, epoch, supp = TrainModel.load_model(config_embed['save_path'] + ".pth", supp=True)
        model_wat = supp["model"]
        # fine tune the watermarked model
        print("First check")
        TrainModel.evaluate(model_wat, test_loader, config_attack)
        model = pruning(model_wat, config_attack)
        print("evaluate the model after fine tuning")
        TrainModel.evaluate(model, test_loader, config_attack)
        extract(model, supp)
        # print_sparsity(model)
        # print(model)

    @staticmethod
    def overwriting_attack(config_embed, config_attack, config_data):
        """embedding """
        # load the watermarked model
        watermarked_model, acc, epoch, supp = TrainModel.load_model(config_embed['save_path'] + ".pth", supp=True)
        model_wat = supp["model"]
        # Get data
        train_loader, test_loader, attack_loader, remind_train_loader, attack_loader_size = \
            Database.load_split_dataset_loaders(config_data)
        # evaluate the model
        print("evaluate the model before watermarking")
        TrainModel.evaluate(model_wat, test_loader, config_embed)
        print("check ber of watermarked model")
        extract(model_wat, supp)
        # changing the batch_size to 1
        # T in the paper
        gmm, x_fc, _, gmm_nonzero = gmm_load(config_attack["n_features"], config_attack["n_components"],
                                             path=config_attack["path_gmm"])
        """Get the model, data and the paths of gmm and the model"""
        watermarked_classes = random.sample([i for i in gmm_nonzero[0]], config_attack["nb_wat_classes"])
        watermarked_classes = torch.tensor(watermarked_classes)
        print("here", watermarked_classes)
        """changing the batch_size"""
        train_data = torch.tensor([]).cuda()
        train_labels = torch.tensor([])
        for idx_batch, data in enumerate(train_loader):
            images, label = data
            images = images.to(config_attack["device"])
            tmp_data = torch.cat([img[None, :] for img in images])
            train_data = torch.cat((train_data, tmp_data), 0)
            train_labels = torch.cat((train_labels, label), 0)

        """generate the trigger set"""
        y_k, key_index, x_key, y_key = get_trigger_set(gmm=gmm, train_data=train_data, train_labels=train_labels,
                                                       watermarked_classes=watermarked_classes, x_fc=x_fc,
                                                       percent=config_attack["percent_ts"])


        """generate the watermark watermark"""
        watermark = Random.get_rand_bits(config_attack["watermark_size"], 0., 1.)
        # watermark = torch.ones(config_embed["watermark_size"])
        watermark = torch.tensor(watermark).reshape(1, config_attack["watermark_size"])
        print("watermark", watermark)
        """generate matrix matrix_a"""
        matrix_a = Random.generate_secret_matrix(config_attack["n_features"], config_attack["watermark_size"])
        """embed the watermark with deepsign_c"""
        """embed the watermark with deepsign_c"""
        model_attacked, ber = embed(train_loader, train_labels, model_wat, test_loader, y_k, x_key, y_key, gmm,
                               matrix_a, watermark,
                               watermarked_classes, config_attack)
        # print("evaluate the watermarked model")
        model_attacked, acc, epoch, supp_attacked = TrainModel.load_model(config_attack['save_path'] + ".pth", supp=True)
        model_attacked = supp_attacked["model"]

        acc = TrainModel.evaluate(model_attacked, test_loader, config_attack)
        # supp["epoch_ext"] = 10
        _, ber_ex = extract(model_attacked, supp)
        print("last ber", ber_ex)
        return acc, ber
