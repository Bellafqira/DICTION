import os
import itertools as it
import random
from copy import deepcopy
import numpy as np
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm

from networks.cnn import CnnModel
from networks.mlp import MLP
from networks.mlp_riga import MLP_RIGA
from networks.resnet import res_net18
from networks.resnet18_two_linear import ResNet18TwoLinear
from torch.utils.data import Dataset

devices = 'cuda' if torch.cuda.is_available() else 'cpu'


class Util:
    @staticmethod
    def hard_th(matrix_g):
        return torch.nn.Threshold(0.5, 0)(matrix_g)

    @staticmethod
    def stack_act(extractor, x_key, config):
        x_fc = torch.cat([extractor(data.to(config["device"]))[config["layer_name"]].detach().cpu() for data, _ in
                          x_key], dim=0)
        return x_fc.cuda()

    @staticmethod
    def _get_act(dict_of_tensors):
        first_key = next(iter(dict_of_tensors))

        # Check if the first tensor has a shape of 4
        if len(dict_of_tensors[first_key].shape) == 4:
            concatenated_tensor_dict = dict_of_tensors[first_key].mean(dim=1)
        else:
            concatenated_tensor_dict = dict_of_tensors[first_key]

        # Reshape the first tensor
        concatenated_tensor_dict = concatenated_tensor_dict.view(concatenated_tensor_dict.shape[0], -1)

        # Loop through the dictionary starting from the second element
        for key in list(dict_of_tensors)[1:]:
            tmp = dict_of_tensors[key]

            # If the tensor has a shape of 4, take the mean along the second dimension
            if len(tmp.shape) == 4:
                tmp = tmp.mean(dim=1)

            # Reshape the tensor
            tmp = tmp.view(tmp.shape[0], -1)

            # Concatenate along the last dimension
            concatenated_tensor_dict = torch.cat((concatenated_tensor_dict, tmp), dim=1)

        return concatenated_tensor_dict


class Random:
    @staticmethod
    def get_rand_bits(size, a, b):
        return random.choices([a, b], k=size)

    @staticmethod
    def select_random_positions(shape, nb_samples):
        if isinstance(shape, int):
            pos_list = list(range(shape))
        else:
            pos = [range(s) for s in shape]
            pos_list = [list(p) for p in it.product(*pos)]
        sampled_list = random.sample(pos_list, nb_samples)
        return sampled_list

    @staticmethod
    def select_random_positions_percent(shape, percent):
        return Random.select_random_positions(shape, int(np.prod(shape) * percent))

    @staticmethod
    def generate_shuffled_list(width, high):
        lst = [list(p) for p in it.product(range(width), range(high))]
        return random.shuffle(lst)

    @staticmethod
    def generate_secret_matrix(width, high):
        # return torch.normal(mean=0, std=1, size=(width, high))
        return torch.randn(size=(width, high))
        # return 2.*torch.rand(size=(width, high)) - 1


class Database:

    @staticmethod
    def get_transforms(database):
        if database == "mnist":
            transform_train = transforms.Compose([
                transforms.ToTensor(), torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))])
            transform_test = transforms.Compose([
                transforms.ToTensor(), torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))])
        elif database == "cifar10":
            transform_train = transforms.Compose([
                transforms.RandomCrop(size=32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        else:
            raise Exception("database doesn't exist")
        return transform_train, transform_test

    @staticmethod
    def get_dataset(database, transform_train, transform_test):
        if database == "mnist":
            train_dataset = torchvision.datasets.MNIST(root='./data',
                                                       train=True,
                                                       transform=transform_train,
                                                       download=True)
            test_dataset = torchvision.datasets.MNIST(root='./data',
                                                      train=False,
                                                      transform=transform_test)
        elif database == "cifar10":
            train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                         train=True,
                                                         transform=transform_train,
                                                         download=True)
            test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                        train=False,
                                                        transform=transform_test)
        else:
            raise Exception("Unknown Database")

        return train_dataset, test_dataset

    @staticmethod
    def get_loaders(database, batch_size):
        """creat a new attack set with same dimension of training set"""
        transform_train, transform_test = Database.get_transforms(database)
        train_dataset, test_dataset = Database.get_dataset(database, transform_train, transform_test)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)
        return train_loader, test_loader

    @staticmethod
    def gen_dataset_loaders(config):
        train_loader, test_loader = Database.get_loaders(database=config["database"], batch_size=config["batch_size"])
        dataset = {"database": config["database"], "train_loader": train_loader, "test_loader": test_loader}
        torch.save(dataset, config["save_path"])
        print("train and test loaders have been created and saved successfully ")

    @staticmethod
    def load_dataset_loaders(config):
        dataset = torch.load(config["save_path"])
        print("loading the following database... ", config["database"])
        return dataset["train_loader"], dataset["test_loader"]


class TrainModel:
    @staticmethod
    def fine_tune(init_model, train_loader, test_loader, config):

        model = deepcopy(init_model)
        model = model.to(config["device"])
        model.train()

        criterion = config["criterion"]
        optimizer = TrainModel.get_optimizer(model, config)
        scheduler = TrainModel.get_scheduler(optimizer, config)

        best_acc = 0
        best_model = deepcopy(model)
        best_model, best_acc, acc = TrainModel.check_acc(model=model, best_model=best_model, test_loader=test_loader,
                                                         config=config,
                                                         best_acc=best_acc, epoch=0)

        acc_list = []
        if config["show_acc_epoch"]:
            acc_list = [acc]

        for param in model.parameters():
            param.requires_grad = True

        for epoch in range(config["epochs"]):
            train_loss = correct = total = 0
            loop = tqdm(train_loader, leave=True)
            for batch_idx, (inputs, targets) in enumerate(loop):
                inputs, targets = inputs.to(config["device"]), targets.to(config["device"])
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                # back propagate the loss
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                # update the progress bar
                loop.set_description(f"Epoch [{epoch}/{config['epochs']}]")
                loop.set_postfix(loss=train_loss / (batch_idx + 1), acc=100. * correct / total,
                                 correct_total=f"[{correct}"
                                               f"/{total}]")
            scheduler.step()
            best_model, best_acc, acc = TrainModel.check_acc(model=model, best_model=best_model,
                                                             test_loader=test_loader,
                                                             config=config,
                                                             best_acc=best_acc, epoch=epoch)

            if config["show_acc_epoch"]:
                acc_list.append(acc)

        if config["show_acc_epoch"]:
            return model, acc_list
        else:
            return model

    @staticmethod
    def get_scheduler(optimizer, cf):
        if cf["scheduler"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cf["milestones"], gamma=cf["gamma"])
        return scheduler

    @staticmethod
    def check_acc(model, best_model, test_loader, config, best_acc, epoch):
        acc = TrainModel.evaluate(model, test_loader, config)
        if acc > best_acc:
            TrainModel.save_model(model, acc, epoch, config["save_path"])  # in config file we use save_path
            best_model = deepcopy(model)
            best_acc = acc
        return best_model, best_acc, acc

    @staticmethod
    def evaluate(init_model, test_loader, config):
        test_loss = correct = total = 0
        loop = tqdm(test_loader, leave=True)
        model = deepcopy(init_model)
        # model.eval()
        criterion = config["criterion"]
        device = config["device"]

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loop):
                inputs, targets = inputs.to(device), targets.to(device)
                # targets = targets.type(torch.long)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                # update the progress bar
                loop.set_description(f"Testing set ")
                loop.set_postfix(loss=test_loss / (batch_idx + 1), acc=f"{100. * (correct / total)}",
                                 correct_total=f"[{correct}"f"/{total}]")
        acc = 100. * correct / total
        return acc

    @staticmethod
    def save_model(model, acc, epoch, path, supplementary=None):
        print('Saving model...')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'supplementary': supplementary
        }
        savedir = os.path.dirname(path)
        if not (os.path.exists(savedir)):
            os.makedirs(savedir)
        torch.save(state, path)

    @staticmethod
    def load_model(path):
        print('Loading model...')
        state = torch.load(path)
        return state

    @staticmethod
    def select_architecture(config_data, config_model):
        """select the suitable architecture"""
        """get train and test loader"""
        train_loader, test_loader = Database.get_loaders(database=config_data["database"],
                                                         batch_size=config_data["batch_size"])
        """load the trained model in the given path"""
        model = TrainModel.load_model(config_model["path_model"])['net']
        return model, train_loader, test_loader

    @staticmethod
    def get_model(architecture, device):
        if architecture == "CNN":
            # ep50, bs =512, lr=0.001, Adam
            model = CnnModel().to(device)
        elif architecture == "MLP":
            # ep50, bs =512, lr=0.01, opt=Adam
            model = MLP().to(device)
        # elif architecture == "ResNet18":
        #     model = res_net18().to(device)
        elif architecture == "ResNet18":
            model = ResNet18TwoLinear().to(device)
        elif architecture == "MLP_RIGA":
            model = MLP_RIGA().to(device)
        else:
            raise Exception("architecture doesn't exist")
        return model

    @staticmethod
    def get_optimizer(model, config, parameters=None):
        if parameters is not None:
            return TrainModel._get_optimizer(config, list(model.parameters()) + list(parameters))
        else:
            return TrainModel._get_optimizer(config, model.parameters())

    @staticmethod
    def _get_optimizer(config, parameters):
        if config["opt"] == "SGD":
            optimizer = torch.optim.SGD(parameters, lr=config["lr"], momentum=config["momentum"],
                                        weight_decay=config["wd"])
        elif config["opt"] == "Adam":
            optimizer = torch.optim.Adam(parameters, lr=config["lr"], weight_decay=config["wd"])
        elif config["opt"] == "RMSprop":
            optimizer = torch.optim.RMSprop(parameters, lr=config["lr"], alpha=0.9, eps=1e-08,
                                            weight_decay=config["wd"],
                                            momentum=config["momentum"], centered=False)
        else:
            raise Exception("Unknown optimizer")
        return optimizer


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, input_tensor):
        return torch.randn(input_tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class AddWhiteSquareTransform:
    def __init__(self, square_size=20, start_x=30, start_y=40):
        self.square_size = square_size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, x):
        # Assuming x is a single image tensor of shape (C, H, W)
        C, H, W = x.size()
        white_square = torch.ones((C, self.square_size, self.square_size))
        if self.start_x + self.square_size <= H and self.start_y + self.square_size <= W:
            x[:, self.start_x:self.start_x + self.square_size,
            self.start_y:self.start_y + self.square_size] = 1  # white_square
        else:
            raise ValueError("Square position and size exceed image dimensions.")
        return x


class CustomTensorDataset(Dataset):
    def __init__(self, x_tensor, y_tensor, transform_list=None):
        self.x_tensor = x_tensor
        self.y_tensor = y_tensor
        self.transforms = transform_list

    def __getitem__(self, index):
        x = self.x_tensor[index]
        y = self.y_tensor[index]
        if self.transforms:
            x = self.transforms(x)
        return x, y

    def __len__(self):
        return len(self.x_tensor)
