import torch.nn as nn

from configs.cf_data.cf_data import cf_mnist_data

config_data = cf_mnist_data

database = config_data["database"]
batch_size = config_data["batch_size"]
device = config_data["device"]

criterion = nn.CrossEntropyLoss()
epochs = 50
architecture = "MLP"

cf_mlp_dict = {"lr": 1e-4, "epochs": epochs, "wd": 0, "opt": "Adam", "batch_size": batch_size,
               "architecture": architecture, "milestones": [25, 45], "gamma": 0.1, "criterion": criterion,
               "scheduler": "MultiStepLR",
               "device": device,
               "database": database,
               "momentum": 0,
               "save_path": "results/trained_models/" + architecture.lower() + "/_db" + database +
                            "_ep" + str(epochs) + "_bs" + str(batch_size) + ".pth",
               "save_fig_path": "results/trained_models/" + architecture.lower() + "/_db" + database
                                + "_ep" + str(epochs) + "_bs" + str(batch_size),
               "show_acc_epoch": True
               }
