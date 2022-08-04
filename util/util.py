import itertools as it
import random
from copy import deepcopy
from math import floor
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from scipy.stats import ortho_group
from torchvision import transforms
from tqdm import tqdm

from networks.cnn import CnnModel
from networks.mlp import MLP
from networks.resnet import res_net18
from networks.wide_resnet import wide_resnet28
from scipy.linalg import orth

devices = 'cuda' if torch.cuda.is_available() else 'cpu'


class Util:
    @staticmethod
    def gen_orthogonal_matrix(size):
        return ortho_group.rvs(size)

    @staticmethod
    def plt_fig(rng, ber_list, acc_list, cf):
        """This function plot the ber and acc  of change sign attack in function of WMR"""
        cols = ['red', 'black']  # ['red', 'blue', 'green', 'Black', 'olive']
        # style = ["s", "o", "x", "*", "2"]
        label = ['BER_WoF', 'ACC_WoF', 'BER_WF', 'ACC_WF']

        for i, (ber_l, acc_l) in enumerate(zip(ber_list, acc_list)):
            plt.scatter(rng, ber_l, c=cols[i], marker='*', label=label[2 * i], s=50)
            plt.scatter(rng, acc_l, c=cols[i], marker='s', label=label[2 * i + 1], s=50)
            for a, ac, br in zip(rng, acc_l, ber_l):
                plt.text(a + 0.04, ac, str(ac))
                plt.text(a + 0.04, br, str(br))

        plt.xticks(np.arange(-1, cf["partitions"] + 2, 1))
        plt.yticks(np.arange(-10, 110, 10))

        plt.legend()  # loc='upper right'
        plt.grid(True)
        plt.xlabel(cf["x_label"])
        plt.ylabel(cf["y_label"])
        plt.title(
            'target = {},  partitions ={}, epochs = {}, attacked_layer = {}'.format(cf["target"], cf["partitions"],
                                                                                    cf['epochs'], cf["layer_name"]))
        # config_embed['save_fig_path'] = f"results/attacks/change50/uchida/{config_embed['architecture'].lower()}/_{config_embed['partitions']}" \
        #                       f"_rec{config_embed['epochs']}_{config_embed['target']}.png"
        print(cf['save_fig_path'])
        plt.savefig(cf['save_fig_path'])
        plt.show()

    @staticmethod
    def dm_qim_embed(mess, bit, delta, prng):
        floor_x = np.floor((mess + bit + prng) / delta)
        if floor_x % 2 == bit:
            qim_support = delta * floor_x - prng
        else:
            qim_support = delta * (floor_x + 1) - prng
        return qim_support

    @staticmethod
    def dm_qim_extract(wat_mess, delta, prng):
        return int(np.floor((wat_mess + (delta / 2) + prng) / delta) % 2)

    @staticmethod
    def dm_qim_embed_list(mess, watermark, delta, prng):
        if len(mess) < len(watermark):
            return ["error"]
        wat_mess = []
        for i, bit in enumerate(watermark):
            wat_mess.append(Util.dm_qim_embed(mess[i], bit, delta, prng[i]))
        return wat_mess

    @staticmethod
    def dm_qim_extract_list(wat_mess, delta, prng):
        return [Util.dm_qim_extract(wat_mess[i], delta, prng[i]) for i in range(len(wat_mess))]

    @staticmethod
    def get_mean_and_std(dataset):
        """Compute the mean and std value of dataset."""
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
        mean = torch.zeros(3)
        std = torch.zeros(3)
        print('==> Computing mean and std..')
        for inputs, targets in dataloader:
            for i in range(3):
                mean[i] += inputs[:, i, :, :].mean()
                std[i] += inputs[:, i, :, :].std()
        mean.div_(len(dataset))
        std.div_(len(dataset))
        return mean, std

    @staticmethod
    def hard_th(matrix_g):
        return torch.nn.Threshold(0.5, 0)(matrix_g)


class Random:
    @staticmethod
    def get_rand_bits(size, a, b):
        return random.choices([a, b], k=size)

    @staticmethod
    def get_rand_bits_delta(size, delta):
        return [random.uniform(- delta / 2, delta / 2) for _ in range(size)]

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
    def load_database(database, batch_size, attacked_class=None):
        transform_train, transform_test = Database.get_transforms(database)
        train_dataset, test_dataset = Database.get_dataset(database, transform_train, transform_test)
        if attacked_class:
            # train_dataset.targets = [attacked_class[1] if x == attacked_class[0] and  else x for x
            #                          in train_dataset.targets]
            positions = [i for i, x in enumerate(train_dataset.targets) if x == attacked_class[0]]
            perc = int(len(positions)*attacked_class[2])
            positions = np.random.permutation(positions)
            pos_a = positions[:perc]
            pos_na = positions[perc:]
            for x in pos_a:
                train_dataset.targets[x] = attacked_class[1]
            vec = []
            for i in range(len(train_dataset.targets)):
                if i in pos_a:
                    vec.append(1)
                elif i in pos_na:
                    vec.append(2)
                else:
                    vec.append(0)
            vec = [vec[i:i+batch_size] for i in range(0, len(train_dataset.targets), batch_size)]
            print(f'the count of {attacked_class[0]} is {train_dataset.targets.count(attacked_class[0])}')

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False, num_workers=2)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False, num_workers=2)

        if attacked_class:
            return train_loader, test_loader, np.array(vec)
        return train_loader, test_loader

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
    def split_dataset(database, batch_size, split_set_size_percent):
        """creat a new attack set with same dimension of training set"""
        transform_train, transform_test = Database.get_transforms(database)
        train_dataset, test_dataset = Database.get_dataset(database, transform_train, transform_test)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False)
        if 0 < split_set_size_percent < 100:
            #  Splitting the dataset
            split_set_size = floor(split_set_size_percent * len(train_dataset) / 100)
            train_split_set = torch.utils.data.random_split(deepcopy(train_dataset),
                                                            [split_set_size, len(train_dataset) -
                                                             split_set_size])
            # pushing the datasets in the loaders

            split_loader = torch.utils.data.DataLoader(dataset=train_split_set[0],
                                                       batch_size=batch_size,
                                                       shuffle=True)
            remind_train_loader = torch.utils.data.DataLoader(dataset=train_split_set[1],
                                                              batch_size=batch_size,
                                                              shuffle=True)
        elif split_set_size_percent == 0:
            split_loader = None
            remind_train_loader = train_loader
        else:
            raise Exception("split_set_size_percent should be in [0, 100[")
        return train_loader, test_loader, split_loader, remind_train_loader

    @staticmethod
    def gen_split_dataset_loaders(config):
        train_loader, test_loader, attack_loader, remind_train_loader = Database.split_dataset(
            database=config["database"],
            batch_size=config["batch_size"],
            split_set_size_percent=config["attack_set_size"])
        dataset = {"database": "mnist", "attack_loader": attack_loader, "train_loader": train_loader,
                   "test_loader": test_loader, "remind_train_loader": remind_train_loader,
                   "attack_loader_size": config["attack_set_size"]}
        torch.save(dataset, f"attack_sets/{config['database']}/_as{config['attack_set_size']}_bs{config['batch_size']}"
                            f".pth")
        print("attack, training, testing sets have been created successfully ")

    @staticmethod
    def load_split_dataset_loaders(config):
        dataset = torch.load(config["save_path"]+".pth")
        return dataset["train_loader"], dataset["test_loader"], dataset["attack_loader"], \
               dataset["remind_train_loader"], dataset["attack_loader_size"]


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
    def evaluate(model, test_loader, config):
        test_loss = correct = total = 0
        loop = tqdm(test_loader, leave=True)
        model.eval()
        criterion = config["criterion"]
        device = config["device"]

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loop):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                # update the progress bar
                loop.set_description(f"Testing set ")
                loop.set_postfix(loss=test_loss / (batch_idx + 1), acc=100. * correct / total,
                                 correct_total=f"[{correct}"f"/{total}]")
        acc = 100. * correct / total
        return acc

    @staticmethod
    def save_model(model, acc, epoch, path, supplementary=None):
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'supplementary': supplementary
        }
        torch.save(state, path + '.pth')

    @staticmethod
    def load_model(path, supp=False):
        dictionary = torch.load(path)
        model = dictionary['net']
        acc = dictionary['acc']
        epoch = dictionary['epoch']
        if supp:
            supplementary = dictionary['supplementary']
            return model, acc, epoch, supplementary
        else:
            return model, acc, epoch

    @staticmethod
    def select_architecture(config):
        """select the suitable architecture"""
        """get train and test loader"""
        train_loader, test_loader = Database.load_database(database=config["database"], batch_size=config["batch_size"])
        """load the trained model in the given path"""
        model = TrainModel.get_model(config["architecture"], config["device"])
        model.load_state_dict(TrainModel.load_model(config["path_model"])[0])
        return model, train_loader, test_loader

    @staticmethod
    def get_model(architecture, device):
        if architecture == "CNN":
            # ep50, bs =512, lr=0.001, m=0.9, Adam
            model = CnnModel().to(device)
        elif architecture == "MLP":
            # ep50, bs =512, lr=0.01, opt=Adam
            model = MLP().to(device)
        elif architecture == "ResNet18":
            model = res_net18().to(device)
            # model = torch.nn.DataParallel(model)
        elif architecture == "Wide_ResNet28":
            model = wide_resnet28().to(device)
            # model = torch.nn.DataParallel(model)
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
