# Ref : https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

import torch.nn as nn
from configs.cf_data.cf_data import cf_cifar10_data

config_data = cf_cifar10_data

device = config_data["device"]
database = config_data["database"]
batch_size = config_data["batch_size"]

epochs = 200
criterion = nn.CrossEntropyLoss()
architecture = "ResNet18"

cf_resnet18_dict = {"batch_size": batch_size, "lr": 0.1, "epochs": epochs, "wd": 5e-4, "opt": "SGD",
                    "architecture": architecture, "milestones": [150, 200], "gamma": 0.1, "criterion": criterion,
                    "scheduler": "CosineAnnealingLR",
                    "device": device,
                    "database": database,
                    "momentum": 0.9,
                    "save_path": "results/trained_models/resnet18/_db" + database + "_ep" + str(epochs) +
                                 "_bs" + str(batch_size),
                    "save_fig_path": "results/trained_models/resnet18/_db" + database
                                     + "_ep" + str(epochs) + "_bs" + str(batch_size),
                    "show_acc_epoch": True
                    }
