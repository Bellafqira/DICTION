import random

import numpy as np
import torch
import time

from matplotlib import pyplot as plt
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

from attacks.pruning import pruning, print_sparsity
from util.util import TrainModel, Random, Database, Util

from watermark.deepsigns import gmm_compute, gmm_load, get_trigger_set, extract, embed


class Tests:

    @staticmethod
    def gmm(config_embed, config_data):
        """computing the gmm"""
        """Get the model and data """
        # Get model
        init_model = TrainModel.get_model(config_embed["architecture"], config_embed["device"])
        init_model.load_state_dict(TrainModel.load_model(config_embed["path_model"])['net'])
        # Get data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        # evaluate the model
        print("Evaluation of the model to watermark")
        TrainModel.evaluate(init_model, test_loader, config_embed)
        # here we save the activation maps of all data in  the tensor x_fc
        # we fit then the GMM to this tensor
        # define extractor
        extractor = create_feature_extractor(init_model, [config_embed["layer_name"]])

        x_fc = Util.stack_x_fc(extractor, train_loader, config_embed)
        print("x_fc", x_fc.shape)
        # """compute and save the gmm"""
        print("start gmm")
        start = time.time()
        gmm_compute(x_fc=x_fc, n_features=config_embed["n_features"], n_components=config_embed["n_components"],
                    path=config_embed["path_gmm"])
        print("GMM computed in  ", time.time() - start, "second")
        print("start loading the GMM")
        gmm, x_fc1, gmm_hist, gmm_nonzero = gmm_load(config_embed["n_features"], config_embed["n_components"],
                                                     path=config_embed["path_gmm"])
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
        init_model.load_state_dict(TrainModel.load_model(config_embed["path_model"])['net'])
        # Get data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        """evaluate the model"""
        TrainModel.evaluate(init_model, test_loader, config_embed)
        # define extractor
        extractor = create_feature_extractor(init_model, [config_embed["layer_name"]])

        # T in the paper
        gmm, _, _, gmm_nonzero = gmm_load(config_embed["n_features"], config_embed["n_components"],
                                          path=config_embed["path_gmm"])
        """Get the model, data and the paths of gmm and the model"""
        watermarked_classes = random.sample([i for i in gmm_nonzero[0]], config_embed["nb_wat_classes"])
        watermarked_classes = torch.tensor(watermarked_classes)
        print("watermarked_classes ", watermarked_classes)

        """Get data, labels and activation maps"""
        train_data = torch.tensor([]).cuda()
        x_fc = torch.tensor([])
        train_labels = torch.tensor([])
        for data in train_loader:
            images, label = data
            images = images.to(config_embed["device"])
            train_data = torch.cat((train_data, images), 0)
            train_labels = torch.cat((train_labels, label), 0)
            x_fc_tmp = extractor(images.to(config_embed["device"]))[config_embed["layer_name"]].detach().cpu()
            x_fc = torch.cat((x_fc, x_fc_tmp), 0)
        print(train_data.shape)
        print(train_labels.shape)
        print(x_fc.shape)

        """generate the trigger set"""
        x_key, y_key = get_trigger_set(gmm=gmm, train_data=train_data, train_labels=train_labels,
                                       watermarked_classes=watermarked_classes, x_fc=x_fc,
                                       percent=config_embed["percent_ts"])

        """embed the watermark with deepsigns"""
        model_wat, ber = embed(train_loader, init_model, test_loader, x_key, y_key, gmm, watermarked_classes,
                               config_embed)

        acc = TrainModel.evaluate(model_wat, test_loader, config_embed)
        return acc, ber

    # @staticmethod
    # def fine_tune_attack(config_embed, config_attack, config_data):
    #     # Get data
    #     train_loader, test_loader = Database.load_dataset_loaders(config_data)
    #     # load the watermarked model
    #     dict_model = TrainModel.load_model(config_embed['save_path'])
    #     model_wat = dict_model["supplementary"]["model"]
    #     # fine tune the watermarked model
    #     print("First check")
    #     TrainModel.evaluate(model_wat, test_loader, config_attack)

    #     for i in range(1, 4):
    #         model_wat = TrainModel.fine_tune(model_wat, train_loader, test_loader, config_attack)
    #         # check the accuracy and BER
    #         print(f"BER after finetuning {i * 50}")
    #         extract(model_wat, dict_model["supplementary"])
    #         TrainModel.evaluate(model_wat, test_loader, config_attack)

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
        _, ber = extract(init_model, dict_model["supplementary"])
        print("BER = ", ber)
        print("Start fine-tuning")
        results_acc = []
        results_ber = []
        epochs = [50, 100, 150]
        for ep in epochs:
            model_wat = TrainModel.fine_tune(model_wat, train_loader, test_loader, config_attack)
            # check the accuracy and BER
            _, ber = extract(model_wat, dict_model["supplementary"])
            acc = TrainModel.evaluate(model_wat, test_loader, config_attack)
            print(f"ACC and BER after finetuning {ep}")
            print("BER = ", ber, "ACC = ", acc)
            results_acc.append(acc)
            results_ber.append(ber)
        print("epochs = ", epochs)
        print("results_acc = ", results_acc)
        print("results_ber = ", results_ber)

    @staticmethod
    def pruning_attack(config_embed, config_attack, config_data):
        # Get data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        # load the watermarked model
        dict_model = TrainModel.load_model(config_embed['save_path'])
        model_wat = dict_model["supplementary"]["model"]
        # fine tune the watermarked model
        print("First check before pruning")
        TrainModel.evaluate(model_wat, test_loader, config_attack)
        results_acc = []
        results_ber = []
        pruning_rate = [x / 10 for x in range(10)] + [0.95, 0.99, 1.]

        for rate in pruning_rate:
            config_attack["amount"] = rate
            model = pruning(model_wat, config_attack)
            print("evaluate the model after pruning of amount", rate)
            results_acc.append(TrainModel.evaluate(model, test_loader, config_attack))
            results_ber.append(extract(model, dict_model["supplementary"])[1])
        print("pruning_rate = ", pruning_rate)
        print("results_acc = ", results_acc)
        print("results_ber = ", results_ber)

    @staticmethod
    def overwriting_attack(config_embed, config_attack, config_data):
        """embedding """
        # Get data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        # load the watermarked model
        dict_model = TrainModel.load_model(config_embed['save_path'])
        model_wat = dict_model["supplementary"]["model"]
        # evaluate the model
        print("evaluate the model before watermarking")
        TrainModel.evaluate(model_wat, test_loader, config_embed)
        # define extractor
        extractor = create_feature_extractor(model_wat, [config_embed["layer_name"]])
        print("check ber of watermarked model")
        extract(model_wat, dict_model["supplementary"])
        # changing the batch_size to 1
        # T in the paper
        gmm, _, _, gmm_nonzero = gmm_load(config_attack["n_features"], config_attack["n_components"],
                                          path=config_attack["path_gmm"])
        """Get the model, data and the paths of gmm and the model"""
        watermarked_classes = random.sample([i for i in gmm_nonzero[0]], config_attack["nb_wat_classes"])
        watermarked_classes = torch.tensor(watermarked_classes)
        print("watermarked classes = ", watermarked_classes)

        """Get data, labels and activation maps"""
        train_data = torch.tensor([]).cuda()
        x_fc = torch.tensor([])
        train_labels = torch.tensor([])
        for data in train_loader:
            images, label = data
            images = images.to(config_embed["device"])
            train_data = torch.cat((train_data, images), 0)
            train_labels = torch.cat((train_labels, label), 0)
            x_fc_tmp = extractor(images.to(config_embed["device"]))[config_embed["layer_name"]].detach().cpu()
            x_fc = torch.cat((x_fc, x_fc_tmp), 0)
        print(train_data.shape)
        print(train_labels.shape)
        print(x_fc.shape)

        """generate the trigger set"""
        x_key, y_key = get_trigger_set(gmm=gmm, train_data=train_data, train_labels=train_labels,
                                       watermarked_classes=watermarked_classes, x_fc=x_fc,
                                       percent=config_attack["percent_ts"])

        """embed the watermark with deepsigns"""
        model_attacked, ber = embed(train_loader, model_wat, test_loader, x_key, y_key, gmm, watermarked_classes,
                                    config_attack)

        # load the watermarked model
        dict_model_att = TrainModel.load_model(config_attack['save_path'])
        model_attacked = dict_model_att["supplementary"]["model"]

        acc = TrainModel.evaluate(model_attacked, test_loader, config_attack)
        # supp["epoch_ext"] = 10
        _, ber_ex = extract(model_attacked, dict_model["supplementary"])
        print("last ber", ber_ex)
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
