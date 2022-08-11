import random
from copy import deepcopy

import torch
import time
# from kmeans_pytorch import kmeans, kmeans_predict
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import transforms
from tqdm import tqdm

from attacks.pruning import pruning, print_sparsity
from networks.customTensorDataset import CustomTensorDataset
from networks.piadetector import PiaDetector
from util.metric import Metric
from util.util import TrainModel, Random, Database

from watermark.diction import extract, embed

writer = SummaryWriter()


class Tests:

    @staticmethod
    def embedding(config_embed, config_data, nb_run=0):
        """DICTION embedding """
        """Get the model and data """
        # Get model
        init_model = TrainModel.get_model(config_embed["architecture"], config_embed["device"])
        init_model.load_state_dict(TrainModel.load_model(config_embed["path_model"])['net'])
        # Get data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        # evaluate the model
        print("evaluate the model before watermarking")
        TrainModel.evaluate(init_model, test_loader, config_embed)
        # embed the watermark with DICTION
        model_wat, ber = embed(init_model, test_loader, train_loader, config_embed)
        print("evaluate the watermarked model")
        acc = TrainModel.evaluate(model_wat, test_loader, config_embed)
        return acc, ber

    @staticmethod
    def fine_tune_attack(config_embed, config_attack, config_data):
        # Get data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        # Get model
        init_model = TrainModel.get_model(config_embed["architecture"], config_embed["device"])
        init_model.load_state_dict(TrainModel.load_model(config_embed["path_model"])['net'])
        init_model.eval()
        # load the watermarked model
        dict_model = TrainModel.load_model(config_embed['save_path'])
        model_wat = dict_model["supplementary"]["model"]
        # fine tune the watermarked model
        print("Check the accuracy of the watermarked model")
        TrainModel.evaluate(model_wat, test_loader, config_attack)
        print("Compute the BER from the original model (non watermarked)")
        extract(init_model, dict_model["supplementary"])
        print("Start fine-tuning")
        for i in range(1, 4):
            model_wat = TrainModel.fine_tune(model_wat, train_loader, test_loader, config_attack)
            # check the accuracy and BER
            print(f"BER after finetuning {i * 50}")
            extract(model_wat, dict_model["supplementary"])
            TrainModel.evaluate(model_wat, test_loader, config_attack)

    @staticmethod
    def pruning_attack(config_embed, config_attack, config_data):
        # Get data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        # load the watermarked model
        dict_model = TrainModel.load_model(config_embed['save_path'])
        model_wat = dict_model["supplementary"]["model"]
        # fine tune the watermarked model
        print("First check")
        TrainModel.evaluate(model_wat, test_loader, config_attack)
        # results_acc = []
        # results_ber = []
        for i in range(10):
            config_attack["amount"] = (i + 1) / 10
            model = pruning(model_wat, config_attack)
            print("evaluate the model after pruning of amount", (i + 1) / 10)
            TrainModel.evaluate(model, test_loader, config_attack)
            extract(model, dict_model["supplementary"])
        # print_sparsity(model)
        # print(model)

    @staticmethod
    def overwriting_attack(config_embed, config_attack, config_data):
        """overwriting attack """
        # Get data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        # load the watermarked model
        dict_model = TrainModel.load_model(config_embed['save_path'])
        model_wat = dict_model["supplementary"]["model"]
        # evaluate the model
        print("Evaluate the model before watermarking")
        TrainModel.evaluate(model_wat, test_loader, config_embed)
        print("Check BER of watermarked model")
        extract(model_wat, dict_model["supplementary"])
        # changing the batch_size to 1
        """embed the watermark with DICTION"""
        model_attacked, ber = embed(model_wat, test_loader, train_loader, config_attack)
        # print("evaluate the watermarked model")
        dict_model_attacked = TrainModel.load_model(config_embed['save_path'])
        model_attacked = dict_model_attacked["supplementary"]["model"]

        acc = TrainModel.evaluate(model_attacked, test_loader, config_attack)
        # supp["epoch_ext"] = 10
        _, ber_1 = extract(model_attacked, dict_model_attacked["supplementary"])
        _, ber_2 = extract(model_attacked, dict_model["supplementary"])
        print("last ber", ber_1, ber_2)
        return acc, ber

    @staticmethod
    def show_weights_distribution(config_embed, config_data):
        # Get data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        # Get model
        init_model = TrainModel.get_model(config_embed["architecture"], config_embed["device"])
        init_model.load_state_dict(TrainModel.load_model(config_embed["path_model"])['net'])
        init_model.eval()
        # evaluate the original model
        print("evaluate the original model")
        TrainModel.evaluate(init_model, test_loader, config_embed)
        # load the watermarked model
        dict_model = TrainModel.load_model(config_embed['save_path'])
        model_wat = dict_model["supplementary"]["model"]
        model_wat.eval()
        # evaluate the original model
        print("evaluate the watermarked model")
        TrainModel.evaluate(model_wat, test_loader, config_embed)

        extractor_init = create_feature_extractor(init_model, [config_embed["layer_name"]])
        extractor_wat = create_feature_extractor(model_wat, [config_embed["layer_name"]])

        x_train, _ = next(iter(train_loader))
        x_fc_init = extractor_init(x_train.cuda())[config_embed["layer_name"]]
        x_fc_wat = extractor_wat(x_train.cuda())[config_embed["layer_name"]]
        x_fc_init = torch.mean(x_fc_init, dim=0)
        x_fc_wat = torch.mean(x_fc_wat, dim=0)

        bins = list(np.arange(-2, 4, 0.2))

        plt.subplot(1, 4, 1)
        plt.hist(torch.flatten(x_fc_init.data).cpu().numpy(), bins=bins, align='mid', edgecolor='red')
        plt.xlabel('Bins')
        plt.ylabel('Frequency')
        plt.title('non watermarked activation maps')
        plt.subplot(1, 4, 2)
        plt.hist(torch.flatten(x_fc_wat.data).cpu().numpy(), bins=bins, align='mid', edgecolor='red')
        plt.xlabel('Bins')
        # plt.ylabel('Frequency')
        plt.title('watermarked activation maps')

        bins = list(np.arange(-1, 2, 0.1))
        for (name_init, param_init), (name, param) in zip(init_model.named_parameters(), model_wat.named_parameters()):
            print(name_init)
            if name == config_embed["layer_name"] + '.weight' or name == "layer4.1.conv2.weight":
                plt.subplot(1, 4, 3)
                plt.hist(torch.flatten(param_init.data).cpu().numpy(), bins=bins, align='mid', edgecolor='red')
                plt.xlabel('Bins')
                # plt.ylabel('Frequency')
                plt.title('non watermarked weights')
                plt.subplot(1, 4, 4)
                plt.hist(torch.flatten(param.data).cpu().numpy(), bins=bins, align='mid', edgecolor='red')
                plt.xlabel('Bins')
                # plt.ylabel('Frequency')
                plt.title('watermarked weights')
                plt.show()
                break

    @staticmethod
    def pia_attack(config_data, config_embed, config_attack):
        # Train PIA Detector on 500 watermarked and 500 not watermarked models
        # And 200 models for testing

        # ************** Generate 1200 models for training and testing **************
        # # Get data
        # train_loader, test_loader = Database.load_dataset_loaders(config_data)
        # # get non watermarked model
        # # Get model
        # init_model = TrainModel.get_model(config_embed["architecture"], config_embed["device"])
        # init_model.load_state_dict(TrainModel.load_model(config_embed["path_model"])['net'])
        # # fine tune models
        # model_ft = deepcopy(init_model)
        # results = {"model_ft": {}, "model_ft_wat": {}}
        # for i in range(config_attack["nb_examples"]):
        #     model_ft = TrainModel.fine_tune(model_ft, train_loader, test_loader, config_attack["train_non_watermarked"])
        #     model_wat, ber = embed(model_ft, test_loader, train_loader, config_attack["train_watermarked"])
        #     # check the accuracy and BER
        #     print(f"BER of finetuning {i}", ber)
        #     results["model_ft"][str(i)] = model_ft
        #     results["model_ft_wat"][str(i)] = model_wat
        #
        # torch.save(results, config_attack['save_path'])

        # ************** Train the detector **************
        # load models and prepare the loaders
        nb_param = 100000

        results = torch.load(config_attack['save_path'])

        weights_ft = torch.cat([results["model_ft"][i].fc2.weight.data.flatten(0)[:nb_param].unsqueeze(0) for i in
                                results["model_ft"].keys()])
        labels_ft = torch.ones(size=(len(results["model_ft"].keys()), 1))

        weights_wat = torch.cat([results["model_ft_wat"][i].fc2.weight.data.flatten(0)[:nb_param].unsqueeze(0)
                                 for i in results["model_ft_wat"].keys()])
        labels_wat = torch.zeros(size=(len(results["model_ft_wat"].keys()), 1))

        (train_data_ft, test_data_ft) = torch.split(weights_ft, [700, 100])
        (train_labels_ft, test_labels_ft) = torch.split(labels_ft, [700, 100])

        (train_data_wat, test_data_wat) = torch.split(weights_wat, [700, 100])
        (train_labels_wat, test_labels_wat) = torch.split(labels_wat, [700, 100])

        train_data = torch.cat((train_data_ft, train_data_wat))
        train_labels = torch.cat((train_labels_ft, train_labels_wat))

        test_data = torch.cat((test_data_ft, test_data_wat))
        test_labels = torch.cat((test_labels_ft, test_labels_wat))

        print(train_data.shape, train_labels.shape)
        print(test_data.shape, test_labels.shape)
        #
        transform_train = transforms.Compose([
            # transforms.RandomCrop(size=32, padding=4),
            # transforms.RandomHorizontalFlip(),
            # AddGaussianNoise(config["mean"], config["std"]),
        ])
        dataset_key = CustomTensorDataset(train_data, train_labels, transform_list=transform_train)
        key_loader_train = DataLoader(dataset=dataset_key, batch_size=64, shuffle=True)

        dataset_key = CustomTensorDataset(test_data, test_labels, transform_list=transform_train)
        key_loader_test = DataLoader(dataset=dataset_key, batch_size=200, shuffle=True)

        model_detector = PiaDetector(nb_param).to(config_data["device"])

        optimizer = optim.Adam(model_detector.parameters(), lr=1e-1)
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.BCEWithLogitsLoss()
        epochs = 100

        for epoch in range(epochs):
            train_loss = 0
            loop = tqdm(key_loader_train, leave=True)
            epoch_acc = 0
            for batch_idx, (inputs, targets) in enumerate(loop):
                inputs, targets = inputs.to(config_data["device"]), targets.to(config_data["device"])
                # targets = targets.type(torch.long)
                optimizer.zero_grad()
                outputs = model_detector(inputs)
                loss = criterion(outputs, targets)
                # back propagate the loss
                loss.backward()
                optimizer.step()
                acc = Metric.binary_acc(outputs, targets)

                train_loss += loss.item()
                epoch_acc += acc.item()

                # update the progress bar
                loop.set_description(f"Epoch [{epoch}/{epochs}]")
                loop.set_postfix(loss=train_loss / (batch_idx + 1), acc=f"{epoch_acc/(batch_idx + 1):.3f}")


    @staticmethod
    def ftal_attack():
        pass

    @staticmethod
    def rtal_attack():
        pass

    @staticmethod
    def refit_attack():
        pass

    # @staticmethod
    # def overwriting_attack(config_embed, config_attack, config_data):
    #     """embedding """
    #
    #     """Get the model and data """
    #     # Get model
    #     init_model = TrainModel.get_model(config_embed["architecture"], config_embed["device"])
    #     init_model.load_state_dict(TrainModel.load_model(config_embed["path_model"] + ".pth")[0])
    #     # Get data
    #     train_loader, test_loader, attack_loader, remind_train_loader, attack_loader_size = \
    #         Database.load_split_dataset_loaders(config_data)
    #     # evaluate the model
    #     print("evaluate the model before watermarking")
    #     TrainModel.evaluate(init_model, test_loader, config_embed)
    #     # changing the batch_size to 1
    #     train_data = torch.tensor([]).cuda()
    #     train_labels = torch.tensor([])
    #     for idx_batch, data in enumerate(train_loader):
    #         images, label = data
    #         images = images.to(config_embed["device"])
    #         tmp_data = torch.cat([img[None, :] for img in images])
    #         train_data = torch.cat((train_data, tmp_data), 0)
    #         train_labels = torch.cat((train_labels, label), 0)
    #     """generate the watermark watermark"""
    #     watermark = Random.get_rand_bits(config_attack["watermark_size"], 0., 1.)
    #     watermark = torch.tensor(watermark).reshape(1, config_attack["watermark_size"])
    #     print("watermark", watermark)
    #     """generate matrix matrix_a"""
    #     matrix_a = Random.generate_secret_matrix(config_attack["n_features"], config_attack["watermark_size"])
    #     """embed the watermark with deepsign_x"""
    #     model_wat, ber = embed(init_model, test_loader, train_loader, train_data, train_labels, matrix_a, watermark,
    #                            config_attack)
    #     print("evaluate the watermarked model")
    #     acc = TrainModel.evaluate(model_wat, test_loader, config_embed)
    #     return acc, ber
