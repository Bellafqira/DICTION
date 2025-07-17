import torch
from torch import nn, optim
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.models.feature_extraction import create_feature_extractor
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from attacks.pruning import pruning
from attacks.dummy_neurons import neuron_clique, neuron_split
from attacks.distillation import train_student
from networks.piadetector import PiaDetector
from util.metric import Metric
from util.util import TrainModel, Database, Util, CustomTensorDataset

from dummy_neurons0 import neuron_clique, neuron_split


class Tests:
    def __init__(self, method: str, model: str):
        self.model = model
        self.method = method
        match method:
            case "DICTION":
                from watermark.diction import extract, embed
                self.extract = extract
                self.embed = embed
            case "DEEPSIGNS":
                from watermark.deepsigns import extract, embed
                self.extract = extract
                self.embed = embed
            case "UCHIDA":
                from watermark.uchida import extract, embed
                self.extract = extract
                self.embed = embed
            case "RES_ENCRYPT":
                from watermark.res_encrypt import extract, embed
                self.extract = extract
                self.embed = embed
            case "RIGA":
                from watermark.riga import extract, embed
                self.extract = extract
                self.embed = embed
            case "HUFUNET":
                # from watermark.hufunet import extract, embed
                # self.extract = extract
                # self.embed = embed
                pass
            case _:
                raise ValueError(f"method {method} not implemented")

    def embedding(self, config_embed, config_data):
        # Get the original model
        init_model = TrainModel.get_model(config_embed["architecture"], config_embed["device"])
        init_model.load_state_dict(TrainModel.load_model(config_embed["path_model"])['net'])
        # Get the training and testing data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        # evaluate the original model
        print(f"evaluate the original model before watermarking with {self.method}")
        TrainModel.evaluate(init_model, test_loader, config_embed)
        # embed the watermark
        model_wat, ber = self.embed(init_model, test_loader, train_loader, config_embed)
        print(f"evaluate the watermarked model with {self.method}")
        acc = TrainModel.evaluate(model_wat, test_loader, config_embed)
        return acc, ber

    def fine_tune_attack(self, config_embed, config_attack, config_data):
        # Get data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        # Get model
        init_model = TrainModel.get_model(config_embed["architecture"], config_embed["device"])
        init_model.load_state_dict(TrainModel.load_model(config_embed["path_model"])['net'])
        # init_model.eval()
        # load the watermarked model
        dict_model = TrainModel.load_model(config_embed['save_path'])
        model_wat = dict_model["supplementary"]["model"]
        # fine tune the watermarked model
        print("Check the accuracy of the watermarked model")
        TrainModel.evaluate(model_wat, test_loader, config_attack)
        print("Compute the BER from the original model (non watermarked)")
        _, ber = self.extract(init_model, dict_model["supplementary"])
        print("BER = ", ber)
        print("Start fine-tuning")
        results_acc = []
        results_ber = []
        epochs = [config_attack["epochs"]*i for i in range(1, 4)]
        # epochs = [50, 100, 150]
        for ep in epochs:
            model_wat = TrainModel.fine_tune(model_wat, train_loader, test_loader, config_attack)
            # check the accuracy and BER
            _, ber = self.extract(model_wat, dict_model["supplementary"])
            acc = TrainModel.evaluate(model_wat, test_loader, config_attack)
            print(f"ACC and BER after finetuning {ep}")
            print("BER = ", ber, "ACC = ", acc)
            results_acc.append(acc)
            results_ber.append(ber)
        print("epochs = ", epochs)
        print("results_acc = ", results_acc)
        print("results_ber = ", results_ber)

    def pruning_attack(self, config_embed, config_attack, config_data):
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
            model = pruning(model_wat, rate)
            print("evaluate the model after pruning of amount", rate)
            results_acc.append(TrainModel.evaluate(model, test_loader, config_attack))
            results_ber.append(self.extract(model, dict_model["supplementary"])[1])
        print("pruning_rate = ", pruning_rate)
        print("results_acc = ", results_acc)
        print("results_ber = ", results_ber)

    def overwriting_attack(self, config_embed, config_attack, config_data):
        # Get model
        init_model = TrainModel.get_model(config_embed["architecture"], config_embed["device"])
        init_model.load_state_dict(TrainModel.load_model(config_embed["path_model"])['net'])
        # Get data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        # load the watermarked model
        dict_model = TrainModel.load_model(config_embed['save_path'])
        model_wat = dict_model["supplementary"]["model"]
        # evaluate the model
        print("Evaluate the model before watermarking")
        TrainModel.evaluate(model_wat, test_loader, config_embed)
        print("Check BER of watermarked model")
        _, ber = self.extract(model_wat, dict_model["supplementary"])
        print("BER = ", ber)
        # embed the watermark with the overwriting attack
        model_attacked, ber = self.embed(model_wat, test_loader, train_loader, config_attack)
        # evaluate the attacked model
        dict_model_attacked = TrainModel.load_model(config_attack['save_path'])
        model_attacked = dict_model_attacked["supplementary"]["model"]

        acc = TrainModel.evaluate(model_attacked, test_loader, config_attack)

        b_ext_1, ber_1 = self.extract(model_attacked, dict_model_attacked["supplementary"])
        b_ext_2, ber_2 = self.extract(model_attacked, dict_model["supplementary"])
        b_ext_3, ber_3 = self.extract(init_model, dict_model_attacked["supplementary"])
        b_ext_4, ber_4 = self.extract(init_model, dict_model["supplementary"])
        b_ext_5, ber_5 = self.extract(model_wat, dict_model_attacked["supplementary"])
        b_ext_6, ber_6 = self.extract(model_wat, dict_model["supplementary"])

        print(f"BER_1 (attacked model with overwrite projection model): {ber_1}")
        print(f"BER_2 (attacked model with watermark projection model): {ber_2}")
        print(f"BER_3 (original model with overwrite projection model): {ber_3}")
        print(f"BER_4 (original model with watermark projection model): {ber_4}")
        print(f"BER_5 (watermarked model with overwrite projection model): {ber_5}")
        print(f"BER_6 (watermarked model with watermark projection model): {ber_6}")

        # b_ext_1 = tensor_vector_to_image(b_ext_1) # model_over + \theta prime
        # b_ext_2 = tensor_vector_to_image(b_ext_2) # model_over + \theta
        # b_ext_3 = tensor_vector_to_image(b_ext_3) # model_orig + \theta prime
        # b_ext_4 = tensor_vector_to_image(b_ext_4) # model_orig + \theta
        # b_ext_5 = tensor_vector_to_image(b_ext_5) # model_wat + \theta prime
        # b_ext_6 = tensor_vector_to_image(b_ext_6) # model_wat + \theta
        # print(f"list of bers", ber_1, ber_2, ber_3, ber_4, ber_5, ber_6)


        # images = [b_ext_1, b_ext_2, b_ext_3, b_ext_4, b_ext_5, b_ext_6]
        # display_images(images)
        # print("watermark 1,", b_ext_1)
        # print("watermark 2,", b_ext_2)
        # print("watermark 3,", b_ext_3)
        # print("watermark 4,", b_ext_4)

        return acc, ber

    def show_weights_distribution(self, config_embed, config_data):
        # Get data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        # Get model
        model_init = TrainModel.get_model(config_embed["architecture"], config_embed["device"])
        model_init.load_state_dict(TrainModel.load_model(config_embed["path_model"])['net'])

        # Layer to inspect
        if isinstance(config_embed["layer_name"], list):
            layer_name = config_embed["layer_name"][-1]
        else:
            layer_name = config_embed["layer_name"].replace(".weight", "")

        # evaluate the original model
        print("evaluate the original model")
        TrainModel.evaluate(model_init, test_loader, config_embed)

        # load the watermarked model
        model_dict = TrainModel.load_model(config_embed['save_path'])
        model_wat = model_dict["supplementary"]["model"]

        # get trigger set
        # x_key, _ = next(iter(model_dict["supplementary"]["x_key"]))

        # evaluate the original model
        print("evaluate the watermarked model")
        print("layer_name", layer_name)
        if layer_name == "linear":  # in the case of resnet18
            layer_name = "view"
        TrainModel.evaluate(model_wat, test_loader, config_embed)
        extractor_init = create_feature_extractor(model_init, [layer_name])
        extractor_wat = create_feature_extractor(model_wat, [layer_name])

        x_train, _ = next(iter(train_loader))  # x
        # x_train = x_key

        # Get activation distributions
        act_init = extractor_init(x_train.cuda())[layer_name]
        act_wat = extractor_wat(x_train.cuda())[layer_name]
        # Compute the mean of activation maps across the batch dimension
        act_init = torch.mean(act_init, dim=0)
        act_wat = torch.mean(act_wat, dim=0)

        # Flatten the activation maps to simplify histogram computation
        act_init_flat = torch.flatten(act_init.data).cpu().numpy()
        act_wat_flat = torch.flatten(act_wat.data).cpu().numpy()

        # Compute the min and max for dynamic binning
        min_act = min(act_init_flat.min(), act_wat_flat.min())
        max_act = max(act_init_flat.max(), act_wat_flat.max())

        # Generate bins dynamically from min to max with steps
        bin_size = 1  # adjust bin size as needed
        bins = np.arange(min_act, max_act + bin_size, bin_size)

        # Compute activation distribution stats
        act_init_mean, act_init_std = act_init_flat.mean(), act_init_flat.std()
        act_wat_mean, act_wat_std = act_wat_flat.mean(), act_wat_flat.std()

        # Plot distributions
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 11.25))

        # Init activations hist
        ax1.hist(act_init_flat, bins=bins, align='mid', edgecolor='red')
        ax1.set(xlabel='Bins', ylabel='Frequency', title='non watermarked activation maps')
        ax1.text(0.5, 0.9, f"mean : {act_init_mean:.2f} \nstd : {act_init_std:.2f}",
                 horizontalalignment='left',
                 verticalalignment='center',
                 transform=ax1.transAxes)

        # Watermarked activations hist
        ax2.hist(act_wat_flat, bins=bins, align='mid', edgecolor='red')
        ax2.set(xlabel='Bins', title='watermarked activation maps')
        ax2.text(0.5, 0.9, f"mean : {act_wat_mean:.2f} \nstd : {act_wat_std:.2f}",
                 horizontalalignment='left',
                 verticalalignment='center',
                 transform=ax2.transAxes)

# Plot model parameter distributions
        for (name_init, param_init), (name, param) in zip(model_init.named_parameters(), model_wat.named_parameters()):
            if name == layer_name + '.weight' or name == "linear.weight":  # only for ResNet18
                # Flatten the weight tensors for histogram plotting
                weights_init = torch.flatten(param_init.data).cpu().numpy()
                weights_wat = torch.flatten(param.data).cpu().numpy()

                # Calculate statistics
                weights_init_mean, weights_init_std = weights_init.mean(), weights_init.std()
                weights_wat_mean, weights_wat_std = weights_wat.mean(), weights_wat.std()

                # Update global min and max for init and wat models
                min_weights = min(weights_init.min(), weights_wat.min())
                max_weights = max(weights_init.max(), weights_wat.max())

                # Compute bins based on overall min and max
                bins = list(np.arange(min_weights, max_weights, 0.1))

                # Histogram for non-watermarked weights
                ax3.hist(weights_init, bins=bins, align='mid', edgecolor='red')
                ax3.set(xlabel='Bins', ylabel='Frequency', title='Non-watermarked weights')
                ax3.text(0.5, 0.9, f"Mean: {weights_init_mean:.2f}\nStd: {weights_init_std:.2f}",
                         horizontalalignment='left',
                         verticalalignment='center',
                         transform=ax3.transAxes)

                # Watermarked
                ax4.hist(weights_wat, bins=bins, align='mid', edgecolor='red')
                ax4.set(xlabel='Bins', ylabel='Frequency', title='Watermarked weights')
                ax4.text(0.5, 0.9, f"Mean: {weights_wat_mean:.2f}\nStd: {weights_wat_std:.2f}",
                         horizontalalignment='left',
                         verticalalignment='center',
                         transform=ax4.transAxes)
                break

        savedir = os.path.join("results/weights", self.model)
        if not (os.path.exists(savedir)):
            os.makedirs(savedir)
        plt.savefig(os.path.join(savedir, f"{self.method}_{self.model}.png"))
        plt.show()

    def dummy_neurons_attack(self, config_embed, config_attack, config_data):
        train_loader, test_loader = Database.load_dataset_loaders(config_data)

        dict_model = TrainModel.load_model(config_attack['path_model'])
        model_wat = dict_model["supplementary"]["model"]

        print("Check the accuracy of the watermarked model")
        TrainModel.evaluate(model_wat, test_loader, config_embed)

        print("Compute the BER from the original model (non watermarked)")
        _, ber = self.extract(model_wat, dict_model["supplementary"])

        print("Original teacher model:")
        linear_layer_indices = []

        for i, layer in enumerate(model_wat.modules()):
            if isinstance(layer, nn.Linear):
                print(f"{i}. Linear layer: {layer.in_features} -> {layer.out_features}")
                linear_layer_indices.append(i)

        print(f"Positions of nn.Linear layers: {linear_layer_indices}")

        if config_attack["attack_type"] == "neuron_clique":
            # # Test NeuronClique
            print("Testing NeuronClique:")
            print("-" * 30)
            attacked_model = neuron_clique(model=model_wat.to(config_attack["device"]),
                                         layer_name=config_attack["layer_name"], num_dummy=config_attack["num_dummy"])
            #
            for i, layer in enumerate(attacked_model.modules()):
                if isinstance(layer, nn.Linear):
                    print(f"{i}. Linear layer: {layer.in_features} -> {layer.out_features}")
                    linear_layer_indices.append(i)
        else:
            # # Test NeuronClique
            print("Testing NeuronSPLIT:")
            print("-" * 30)
            # new_model = neuron_split(model, layer_name="encoder.mlp.fc1", neuron_idx=3, num_splits=4)
            attacked_model = neuron_split(model=model_wat.to(config_attack["device"]),
                                          layer_name=config_attack["layer_name"], neuron_idx=config_attack["neuron_idx"]
                                          , num_splits=config_attack["num_splits"])
            #

            for i, layer in enumerate(attacked_model.modules()):
                if isinstance(layer, nn.Linear):
                    print(f"{i}. Linear layer: {layer.in_features} -> {layer.out_features}")
                    linear_layer_indices.append(i)

        # I need to save the model
        savedir = os.path.dirname(config_attack['save_path'])
        if not (os.path.exists(savedir)):
            os.makedirs(savedir)

        torch.save(attacked_model, config_attack['save_path'])
        print("Attacked model with " + config_attack["attack_type"] + " performance")
        TrainModel.evaluate(attacked_model.to(config_attack["device"]), test_loader, config_embed)

    def distillation(self, config_embed, config_attack, config_data):
        # start by laoding the data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        # I need to load the watermarked model for having the watermarking parameters
        dict_model = TrainModel.load_model(config_embed['save_path'])
        model_wat = dict_model["supplementary"]["model"]

        print("Check the accuracy of the watermarked model")
        TrainModel.evaluate(model_wat, test_loader, config_embed)

        # I need to load the teacher model that just has been attacked with the dummy neurons and called the teacher
        teacher = torch.load(config_attack['path_model']).to(config_embed["device"])

        # I need to load the student model not watermarked and check if it s not watermarked

        student = TrainModel.get_model(config_embed["architecture"], config_embed["device"])

        student = train_student(student, teacher, train_loader, temperature=2.0, lr=1e-3,
                                epochs=config_attack["epoch_attack"], supp=dict_model["supplementary"], device="cuda",
                      extract=self.extract, layer_name=config_attack["layer_name"])


        torch.save(student, config_attack['save_path'])

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
        #     model_ft = TrainModel.fine_tune(model_ft, train_loader, test_loader,
        #     config_attack["train_non_watermarked"])
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
                loop.set_postfix(loss=train_loss / (batch_idx + 1), acc=f"{epoch_acc / (batch_idx + 1):.3f}")

    @staticmethod
    def gen_database(config_data):
        """ generate a new database"""
        Database.gen_dataset_loaders(config_data)

    @staticmethod
    def train_model(config_data, config_train):
        # get data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        # get model
        init_model = TrainModel.get_model(config_train["architecture"], config_train["device"])
        print("Model to train...")
        print(init_model)
        """Start training the model"""
        if config_train["show_acc_epoch"]:
            _, acc_list = TrainModel.fine_tune(init_model, train_loader, test_loader, config_train)
            Tests.plot_acc(acc_list, config_train)
        else:
            TrainModel.fine_tune(init_model, train_loader, test_loader, config_train)

    @staticmethod
    def plot_acc(acc_list, config_train):
        epochs = np.arange(len(acc_list))
        plt.plot(epochs, acc_list, c="black", marker='*', label=f"ACC of {config_train['architecture']} "
                                                                f"over database = {config_train['database']}")
        # for a, acc in zip(epochs, acc_list):
        #     plt.text(a + 0.02, acc + 0.02, str(acc))
        plt.xticks(np.arange(-1, len(acc_list) + 2, 1))
        plt.yticks(np.arange(-10, 110, 10))
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()
        plt.grid(True)
        plt.title(f"ACC of {config_train['architecture']} over database = {config_train['database']}")
        plt.savefig(config_train['save_fig_path'])
        plt.show()