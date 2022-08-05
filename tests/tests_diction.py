import random
from copy import deepcopy

import torch
import time
# from kmeans_pytorch import kmeans, kmeans_predict
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import os
import matplotlib.pyplot as plt
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from attacks.pruning import pruning, print_sparsity
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
        train_loader, test_loader, attack_loader, remind_train_loader, attack_loader_size = \
            Database.load_split_dataset_loaders(config_data)
        # Get model
        init_model = TrainModel.get_model(config_embed["architecture"], config_embed["device"])
        init_model.load_state_dict(TrainModel.load_model(config_embed["path_model"] + ".pth")[0])
        init_model.eval()
        # load the watermarked model
        watermarked_model, acc, epoch, supp = TrainModel.load_model(config_embed['save_path'] + ".pth", supp=True)
        model_wat = supp["model"]
        extractor_init = create_feature_extractor(init_model, [config_embed["layer_name"]])
        extractor_wat = create_feature_extractor(model_wat, [config_embed["layer_name"]])

        x_train, _ = next(iter(train_loader))
        x_fc_init = extractor_init(x_train.cuda())[config_embed["layer_name"]]
        x_fc_wat = extractor_wat(x_train.cuda())[config_embed["layer_name"]]
        x_fc_init = torch.mean(x_fc_init, dim=0)
        x_fc_wat = torch.mean(x_fc_wat, dim=0)

        # writer.add_histogram('distribution activations', x_fc_init.cpu(), 0)
        # writer.add_histogram('distribution activations', x_fc_wat.cpu(), 1)

        bins = list(np.arange(-2, 2, 0.1))

        plt.subplot(2, 2, 1)
        # hist_init = torch.histc(x_fc_init, min=-2, max=2, bins=len(bins))
        plt.hist(torch.flatten(x_fc_init.data).cpu().numpy(), bins=bins, align='mid', edgecolor='red')
        # plt.bar(bins, hist_init.cpu(), align='center', width=0.1, color=['forestgreen'])
        plt.xlabel('Bins')
        plt.ylabel('Frequency')
        plt.title('Histogram of non watermarked model activation maps')
        plt.subplot(2, 2, 2)
        # hist_wat = torch.histc(x_fc_wat, min=-2, max=2, bins=len(bins))
        plt.hist(torch.flatten(x_fc_wat.data).cpu().numpy(), bins=bins, align='mid', edgecolor='red')
        # plt.bar(bins, hist_wat.cpu(), align='center', width=0.1, color=['forestgreen'])
        # plt.bar(x, hist_init.cpu(), align='center', color=['forestgreen'])
        plt.xlabel('Bins')
        plt.ylabel('Frequency')
        plt.title('Histogram of watermarked model activation maps')

        for (name_init, param_init), (name, param) in zip(init_model.named_parameters(), model_wat.named_parameters()):
            print(name_init)
            if name == 'fc2.weight':
                writer.add_histogram('distribution weights', param.data, 0)
                writer.add_histogram('distribution weights', param_init.data, 1)
                step = 0.1
                # bins = list(np.arange(-1, 1, step))
                bins = 20
                plt.subplot(2, 2, 3)
                # hist_init = torch.histc(param_init.data, min=-0.01, max=0.01, bins=len(bins))
                plt.hist(torch.flatten(param_init.data).cpu().numpy(), bins=bins, align='mid', edgecolor='red')
                # print(hist_init.data)
                # plt.bar(bins, hist_init.cpu(), align='center', width=0.1, color=['forestgreen'])
                plt.xlabel('Bins')
                plt.ylabel('Frequency')
                plt.title('Histogram of non watermarked model weights')
                plt.subplot(2, 2, 4)
                # hist_wat = torch.histc(param.data, min=-0.01, max=0.01, bins=len(bins))
                plt.hist(torch.flatten(param.data).cpu().numpy(), bins=bins, align='mid', edgecolor='red')
                # plt.bar(bins, hist_wat.cpu(), align='center', width=0.1, color=['forestgreen'])
                # plt.bar(x, hist_init.cpu(), align='center', color=['forestgreen'])
                plt.xlabel('Bins')
                plt.ylabel('Frequency')
                plt.title('Histogram of watermarked model weights')
                plt.show()

    @staticmethod
    def pia_attack(config_data, config_embed, config_attack):
        #
        # Train first detector 1024 watermarked and 1024 not watermarked and 100 for testing

        # Generate 1024 non watermarked models
        # Get data
        train_loader, test_loader, attack_loader, remind_train_loader, attack_loader_size = \
            Database.load_split_dataset_loaders(config_data)
        # get non watermarked model
        # Get model
        init_model = TrainModel.get_model(config_embed["architecture"], config_embed["device"])
        init_model.load_state_dict(TrainModel.load_model(config_embed["path_model"] + ".pth")[0])

        # _, _, _, supp = TrainModel.load_model(config_embed['path_model'] + ".pth", supp=True)
        # init_model = supp["model"]

        # fine tune models
        model_ft = deepcopy(init_model)
        results = {}
        results["model_ft"] = {}
        results["model_ft_wat"] = {}
        for i in range(config_attack["nb_examples"]):
            model_ft = TrainModel.fine_tune(model_ft, train_loader, test_loader, config_attack["train_non_watermarked"])
            model_wat, ber = embed(model_ft, test_loader, train_loader, config_attack["train_watermarked"])
            # check the accuracy and BER
            print(f"BER of finetuning {i}", ber)
            results["model_ft"] = {"model_ft" + str(i): model_ft}
            results["model_ft_wat"] = {"model_wat" + str(i): model_wat}

        torch.save(results, config_attack['save_path'] + ".pth")
        # # load the watermarked model
        # watermarked_model, acc, epoch, supp = TrainModel.load_model(config_embed['save_path'] + ".pth", supp=True)
        # model_wat = supp["model"]
        # # fine tune the watermarked model
        # print("First check")
        # TrainModel.evaluate(model_wat, test_loader, config_attack)
        #
        # extract(init_model, supp)
        pass

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
