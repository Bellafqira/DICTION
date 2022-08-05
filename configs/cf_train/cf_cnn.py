import torch.nn as nn
from configs.cf_data.cf_data import cf_cifar10_data

config_data = cf_cifar10_data

database = config_data["database"]
batch_size = config_data["batch_size"]
device = config_data["device"]

criterion = nn.CrossEntropyLoss()
architecture = "CNN"
epochs = 100

cf_cnn_dict = {"batch_size": batch_size, "lr": 1e-3, "epochs": epochs, "wd": 1e-4, "opt": "Adam",
               "architecture": architecture, "milestones": [50], "gamma": 0.1, "criterion": criterion,
               "scheduler": "MultiStepLR",
               "momentum": 0,
               "device": device,
               "database": database,
               "save_path": "results/trained_models/" + architecture.lower() + "/_db" + database + "_ep" + str(epochs)
                            + "_bs" + str(batch_size) + ".pth",
               "save_fig_path": "results/trained_models/" + architecture.lower() + "/_db" + database
                                + "_ep" + str(epochs) + "_bs" + str(batch_size),
               "show_acc_epoch": True}
