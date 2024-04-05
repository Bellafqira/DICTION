from configs.cf_train.cf_mlp import cf_mlp_dict
from configs.cf_train.cf_cnn import cf_cnn_dict
from configs.cf_train.cf_mlp_riga import cf_mlp_riga_dict
from configs.cf_train.cf_resnet18 import cf_resnet18_dict

from copy import deepcopy

# ************* watermarking MLP architecture ****************
watermark_size = 256
epochs_embed = 10000
epoch_check = 30
layer_name = "fc2"
save_path_embed = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epochs_embed) + "_epc" + \
                  str(epoch_check)

cf_mlp_embed = {"configuration": cf_mlp_dict,
                "database": cf_mlp_dict["database"],
                "watermark_size": watermark_size,
                "epochs": epochs_embed,
                "epoch_check": epoch_check,

                # Latent space parameters
                "batch_size": int(cf_mlp_dict["batch_size"] * 0.5),  # the number of samples of trigger set in a batch
                "mean": 0,  # the mean of the latent space
                "std": 1,  # the standard deviation of the latent space
                "square_size": 5,  # the size of the square to be added to the trigger set
                "start_x": 0,  # the x coordinate of the square to be added to the trigger set
                "start_y": 0,  # the y coordinate of the square to be added to the trigger set
                "square_value": 1,  # the value of the square to be added to the trigger set
                "n_features": 0.5,  # the percentage of features to be used in the watermarking process
                "lambda": 1e-0,  # the regularization parameter to train the projection model
                "layer_name": layer_name,  # the layer name to be watermarked

                "lr": 1e-3,  # learning rate to train the target model
                "wd": 0,    # weight decay to train the target model
                "opt": "Adam",  # optimizer
                "scheduler": "MultiStepLR",
                "architecture": cf_mlp_dict["architecture"],
                "momentum": 0,
                "milestones": [100, 1000],
                "gamma": 0.1,
                "criterion": cf_mlp_dict["criterion"],
                "device": cf_mlp_dict["device"],

                "path_model": cf_mlp_dict["save_path"],  # Path of the original model
                "save_path": "results/watermarked_models/diction/" + cf_mlp_dict['architecture'].lower() + "/"
                             + save_path_embed + ".pth"  # Path to save the target/watermarked model

                }

epoch_attack = 50
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + ".pth"

cf_mlp_attack_ft = {"configuration": cf_mlp_dict,
                    "database": cf_mlp_dict["database"],
                    "watermark_size": watermark_size,
                    "epochs": epoch_attack,
                    "lr": 1e-3,
                    "wd": 0,
                    "opt": "Adam",
                    "scheduler": "MultiStepLR",
                    "architecture": cf_mlp_dict["architecture"],

                    "path_model": cf_mlp_embed["save_path"],
                    "save_path": "results/attacks/finetuning/diction/" + cf_mlp_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",
                    "momentum": 0,
                    "milestones": [100, 2000],
                    "gamma": 0,
                    "criterion": cf_mlp_dict["criterion"],
                    "device": cf_mlp_dict["device"],
                    "save_fig_path": "results/attacks/finetuning/diction/" + cf_mlp_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".png",
                    "x_label": "Partition_diction_" + cf_mlp_dict["architecture"].lower(),
                    "y_label": "BER/ACC",
                    "show_acc_epoch": False
                    }

cf_mlp_attack_pr = {"configuration": cf_mlp_dict,
                    "database": cf_mlp_dict["database"],
                    "path_model": cf_mlp_embed["save_path"],
                    "watermark_size": watermark_size,
                    "criterion": cf_mlp_dict["criterion"],
                    "save_path": "results/attacks/pruning/diction/" + cf_mlp_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",
                    "architecture": cf_mlp_dict["architecture"],
                    "device": cf_mlp_dict["device"],
                    }

epoch_attack = 10
watermark_size = 256
epoch_check = 3
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) \
                   + "_epc" + str(epoch_check)
cf_mlp_attack_ow = {"configuration": cf_cnn_dict,
                    "database": cf_mlp_dict["database"],
                    "path_model": cf_mlp_embed["save_path"],
                    "watermark_size": watermark_size,

                    # Latent space parameters
                    "mean": 0.26,
                    "std": 0.1,
                    "n_features": 0.5,
                    "square_size": 5,
                    "start_x": 0,
                    "start_y": 0,
                    "square_value": 1,

                    "batch_size": int(cf_mlp_dict["batch_size"] * 0.5),
                    "epochs": epoch_attack,
                    "epoch_check": epoch_check,

                    # Training parameters
                    "lambda": 1,
                    "layer_name": layer_name,
                    "lr": 1e-3,

                    "wd": 0,
                    "scheduler": "MultiStepLR",
                    "architecture": cf_mlp_dict["criterion"],
                    "save_path": "results/attacks/overwriting/diction/" + cf_mlp_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",
                    "momentum": 0,
                    "milestones": [5, 10],
                    "gamma": 0.1,
                    "criterion": cf_mlp_dict["criterion"],
                    "device": cf_mlp_dict["device"],
                    "save_fig_path": "results/attacks/overwriting/diction/" + cf_mlp_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".png",
                    "x_label": "Partition_diction_" + cf_mlp_dict["architecture"].lower(),
                    "y_label": "BER/ACC",
                    "show_acc_epoch": False
                    }

cf_non_watermarked = deepcopy(cf_mlp_dict)
cf_non_watermarked["epochs"] = 2
cf_non_watermarked["show_acc_epoch"] = False
cf_watermarked = deepcopy(cf_mlp_embed)
cf_watermarked["epochs"] = 10
cf_watermarked["show_acc_epoch"] = False
nb_examples = 800
save_path_attack = "_l" + layer_name + "_wat" + str(
    cf_watermarked["watermark_size"]) + "_ep" + str(cf_non_watermarked["epochs"]) + "_nb_examples" + str(nb_examples)

cf_mlp_attack_pia = {"train_non_watermarked": cf_non_watermarked,
                     "train_watermarked": cf_watermarked,
                     "save_path": "results/attacks/pia/diction/" + cf_mlp_dict[
                         "architecture"].lower() + "/" + save_path_attack + ".pth",
                     "nb_examples": nb_examples,

                     }

# ************* watermarking CNN architecture ****************
watermark_size = 256
epochs_embed = 10000
layer_name = "fc1"
epoch_check = 30
save_path_embed = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epochs_embed) + "_epc" + \
                  str(epoch_check)

cf_cnn_embed = {"configuration": cf_cnn_dict,
                "database": cf_cnn_dict["database"],
                "watermark_size": watermark_size,
                "epochs": epochs_embed,
                "epoch_check": epoch_check,

                # Latent space parameters
                "batch_size": int(cf_cnn_dict["batch_size"] * 0.5),
                "mean": 0,
                "std": 1,

                "square_size": 5,
                "start_x": 0,
                "start_y": 0,
                "square_value": 1,

                "n_features": 0.1,
                "lambda": 1e-0,
                "layer_name": layer_name,

                "lr": 1e-3,
                "wd": 0,
                "opt": "Adam",
                "scheduler": "MultiStepLR",
                "architecture": cf_cnn_dict['architecture'],
                "momentum": 0,
                "milestones": [20, 3000],
                "gamma": .1,
                "criterion": cf_cnn_dict["criterion"],
                "device": cf_cnn_dict["device"],

                "path_model": cf_cnn_dict["save_path"],
                "save_path": "results/watermarked_models/diction/" + cf_cnn_dict['architecture'].lower() + "/"
                             + save_path_embed + ".pth",
                }

epoch_attack = 50
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + "_epc" + \
                   str(epoch_check)

cf_cnn_attack_ft = {"configuration": cf_cnn_dict,
                    "database": cf_cnn_dict["database"],
                    "path_model": cf_cnn_embed['save_path'],
                    "epochs": epoch_attack,
                    "watermark_size": watermark_size,

                    "lr": 1e-4,
                    "wd": 0,
                    "scheduler": "MultiStepLR",
                    "opt": "Adam",
                    "architecture": cf_cnn_dict["architecture"],

                    "momentum": 0,
                    "milestones": [150, 200],
                    "gamma": 0.1,
                    "criterion": cf_cnn_dict["criterion"],
                    "device": cf_cnn_dict["device"],

                    "save_path": "results/attacks/finetuning/diction/" + cf_cnn_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",
                    "save_fig_path": "results/attacks/finetuning/diction/" + cf_cnn_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".png",

                    "x_label": "Partition_diction_CNN",
                    "y_label": "BER/ACC",
                    "show_acc_epoch": False
                    }

cf_cnn_attack_pr = {"configuration": cf_cnn_dict,
                    "database": cf_cnn_dict["database"],
                    "path_model": cf_cnn_embed["save_path"],
                    "watermark_size": watermark_size,
                    "criterion": cf_cnn_dict["criterion"],
                    "architecture": cf_cnn_dict["architecture"],
                    "save_path": "results/attacks/pruning/diction/" + cf_cnn_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",
                    "device": cf_cnn_dict["device"],
                    }

layer_name = "fc1"
epoch_attack = 30
epoch_check = 10
watermark_size = 512
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) \
                   + "_epc" + str(epoch_check)

cf_cnn_attack_ow = {"configuration": cf_cnn_dict,
                    "database": cf_cnn_dict["database"],
                    "path_model": cf_cnn_embed["save_path"],
                    "watermark_size": watermark_size,

                    # Latent space parameters
                    "mean": 0,
                    "std": 1,
                    "square_size": 5,
                    "start_x": 0,
                    "start_y": 0,
                    "square_value": 1,
                    "n_features": 0.2,
                    "batch_size": int(cf_cnn_dict["batch_size"] * 0.5),
                    "epochs": epoch_attack,
                    "epoch_check": epoch_check,

                    # Training parameters
                    "lambda": 1,
                    "layer_name": layer_name,
                    "lr": 1e-4,
                    "wd": 0,
                    "opt": "Adam",
                    "scheduler": "MultiStepLR",
                    "architecture": cf_cnn_dict["criterion"],

                    "momentum": 0,
                    "milestones": [1000, 2000],
                    "gamma": 0,
                    "criterion": cf_cnn_dict["criterion"],
                    "device": cf_cnn_dict["device"],

                    "save_path": "results/attacks/overwriting/diction/" + cf_cnn_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",
                    "save_fig_path": "results/attacks/overwriting/diction/" + cf_cnn_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".png",
                    "x_label": "Partition_diction_" + cf_cnn_dict["architecture"].lower(),
                    "y_label": "BER/ACC",
                    "show_acc_epoch": False
                    }

cf_cnn_attack_pia = {"train_non_watermarked": cf_non_watermarked,
                     "train_watermarked": cf_watermarked,
                     "save_path": "results/attacks/pia/diction/" + cf_cnn_dict[
                         "architecture"].lower() + "/" + save_path_attack + ".pth",
                     "nb_examples": nb_examples
                     }

# ************* watermarking Resnet18 architecture ****************
watermark_size = 256
epochs_embed = 1000
epoch_check = 30
layer_name = "layer3.1.conv1"
# layer1.0.conv1', layer3.1.conv1, view
save_path_embed = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epochs_embed) + "_epc" + \
                  str(epoch_check)

cf_resnet18_embed = {"configuration": cf_resnet18_dict,
                     "database": cf_resnet18_dict["database"],
                     "watermark_size": watermark_size,
                     "epochs": epochs_embed,
                     "epoch_check": epoch_check,
                     # Latent space parameters
                     "mean": 0,
                     "std": 1,
                     "square_size": 5,
                     "start_x": 0,
                     "start_y": 0,
                     "square_value": 1,
                     "batch_size": int(cf_resnet18_dict["batch_size"] * 0.5),
                     "n_features": 0.01,
                     "lambda": 1e-0,
                     "layer_name": layer_name,

                     "lr": 1e-4,
                     "wd": 0,
                     "opt": "Adam",
                     "scheduler": "MultiStepLR",
                     "architecture": cf_resnet18_dict['architecture'],
                     "save_path": "results/watermarked_models/diction/" + cf_resnet18_dict[
                         'architecture'].lower() + "/" + save_path_embed + ".pth",
                     # relu but change in mlp cd
                     "path_model": cf_resnet18_dict["save_path"],
                     "momentum": 0,
                     "milestones": [50, 150],
                     "gamma": .1,
                     "criterion": cf_resnet18_dict["criterion"],
                     "device": cf_resnet18_dict["device"]
                     }

epoch_attack = 50
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + "_epc" + \
                   str(epoch_check)

cf_resnet18_attack_ft = {"configuration": cf_resnet18_dict,
                         "database": cf_resnet18_dict["database"],
                         "path_model": cf_resnet18_embed['save_path'] + '.pth',
                         "epochs": epoch_attack,
                         "watermark_size": watermark_size,

                         "layer_name": layer_name,
                         "lr": 1e-4,
                         "wd": 0,
                         "scheduler": "MultiStepLR",
                         "opt": "Adam",
                         "architecture": cf_resnet18_dict["architecture"],
                         "save_path": "results/attacks/finetuning/diction/" + cf_resnet18_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",
                         "momentum": 0,
                         "milestones": [150, 200],
                         "gamma": 0,
                         "criterion": cf_resnet18_dict["criterion"],
                         "device": cf_resnet18_dict["device"],
                         "save_fig_path": "results/attacks/finetuning/diction/" + cf_resnet18_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".png",
                         "x_label": "Partition_diction_resnet18",
                         "y_label": "BER/ACC",
                         "show_acc_epoch": False
                         }

cf_resnet18_attack_pr = {"configuration": cf_resnet18_dict,
                         "database": cf_resnet18_dict["database"],
                         "path_model": cf_resnet18_embed["save_path"] + ".pth",
                         "watermark_size": watermark_size,
                         "criterion": cf_resnet18_dict["criterion"],
                         "architecture": cf_resnet18_dict["architecture"],
                         "save_path": "results/attacks/pruning/diction/" + cf_resnet18_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",
                         "device": cf_resnet18_dict["device"],
                         }

layer_name = "layer1.0.conv1"
# layer1.0.conv1', layer3.1.conv1, view
epoch_attack = 100
watermark_size = 512
epoch_check = 20
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) \
                   + "_epc" + str(epoch_check)

cf_resnet18_attack_ow = {"configuration": cf_resnet18_dict,
                         "database": cf_resnet18_dict["database"],
                         "path_model": cf_resnet18_embed["save_path"] + ".pth",
                         "watermark_size": watermark_size,
                         # Latent space parameters
                         "mean": 0,
                         "std": 1,
                         "square_size": 5,
                         "start_x": 0,
                         "start_y": 0,
                         "square_value": 1,

                         "n_features": 0.01,
                         "batch_size": int(cf_resnet18_dict["batch_size"] * 0.5),
                         "epochs": epoch_attack,
                         "epoch_check": epoch_check,

                         # Training parameters
                         "lambda": 1,
                         "layer_name": layer_name,
                         "lr": 1e-4,
                         "wd": 0,
                         "opt": "Adam",
                         "scheduler": "MultiStepLR",
                         "architecture": cf_resnet18_dict["criterion"],
                         "save_path": "results/attacks/overwriting/diction/" + cf_resnet18_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",
                         "momentum": 0,
                         "milestones": [1000, 2000],
                         "gamma": 0,
                         "criterion": cf_resnet18_dict["criterion"],
                         "device": cf_resnet18_dict["device"],
                         "save_fig_path": "results/attacks/overwriting/diction/" + cf_resnet18_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".png",
                         "x_label": "Partition_diction_" + cf_resnet18_dict["architecture"].lower(),
                         "y_label": "BER/ACC",
                         "show_acc_epoch": False
                         }

cf_resnet18_attack_pia = {"train_non_watermarked": cf_non_watermarked,
                          "train_watermarked": cf_watermarked,
                          "save_path": "results/attacks/pia/diction/" + cf_resnet18_dict[
                              "architecture"].lower() + "/" + save_path_attack + ".pth",
                          "nb_examples": nb_examples
                          }

# ************* watermarking MLP_RIGA architecture ****************
watermark_size = 256
epochs_embed = 1000
layer_name = "conv1"  # "conv1", "flatten", "fc2"
epoch_check = 30
save_path_embed = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epochs_embed) + "_epc" + \
                  str(epoch_check)

cf_mlp_riga_embed = {"configuration": cf_mlp_riga_dict,
                     "database": cf_mlp_riga_dict["database"],
                     "watermark_size": watermark_size,
                     "epochs": epochs_embed,
                     "epoch_check": epoch_check,

                     # Latent space parameters
                     "batch_size": int(cf_mlp_riga_dict["batch_size"]*0.5),
                     "mean": 0,
                     "std": 1,
                     "square_size": 5,
                     "start_x": 0,
                     "start_y": 0,
                     "square_value": 1,
                     "n_features": 0.5,
                     "lambda": 1,

                     "layer_name": layer_name,
                     "lr": 1e-3,
                     "wd": 0,
                     "opt": "Adam",
                     "scheduler": "MultiStepLR",
                     "architecture": cf_mlp_riga_dict["architecture"],

                     "path_model": cf_mlp_riga_dict["save_path"],
                     "save_path": "results/watermarked_models/diction/" + cf_mlp_riga_dict['architecture'].lower() + "/"
                                  + save_path_embed + ".pth",
                     "momentum": 0,
                     "milestones": [100, 1000],
                     "gamma": 0.1,
                     "criterion": cf_mlp_riga_dict["criterion"],
                     "device": cf_mlp_riga_dict["device"]
                     }

epoch_attack = 50
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + "_epc" + \
                   str(epoch_check)

cf_mlp_riga_attack_ft = {"configuration": cf_mlp_riga_embed,
                         "database": cf_mlp_riga_embed["database"],
                         "watermark_size": watermark_size,
                         "epochs": epoch_attack,
                         "lr": 1e-3,
                         "wd": 0,
                         "opt": "Adam",
                         "scheduler": "MultiStepLR",
                         "architecture": cf_mlp_riga_dict["architecture"],

                         "path_model": cf_mlp_riga_embed["save_path"],
                         "save_path": "results/attacks/finetuning/diction/" + cf_mlp_riga_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",
                         "momentum": 0,
                         "milestones": [100, 2000],
                         "gamma": 0,
                         "criterion": cf_mlp_riga_dict["criterion"],
                         "device": cf_mlp_riga_dict["device"],
                         "save_fig_path": "results/attacks/finetuning/diction/" + cf_mlp_riga_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".png",
                         "x_label": "Partition_diction_" + cf_mlp_riga_dict["architecture"].lower(),
                         "y_label": "BER/ACC",
                         "show_acc_epoch": False
                         }

cf_mlp_riga_attack_pr = {"configuration": cf_mlp_riga_dict,
                         "database": cf_mlp_riga_dict["database"],
                         "path_model": cf_mlp_riga_embed["save_path"],
                         "watermark_size": watermark_size,
                         "criterion": cf_mlp_riga_dict["criterion"],
                         "save_path": "results/attacks/pruning/diction/" + cf_mlp_riga_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",
                         "architecture": cf_mlp_riga_dict["architecture"],
                         "device": cf_mlp_riga_dict["device"],
                         }

epoch_attack = 40
watermark_size = 256
epoch_check = 30
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) \
                   + "_epc" + str(epoch_check)

cf_mlp_riga_attack_ow = {"configuration": cf_mlp_riga_embed,
                         "database": cf_mlp_riga_embed["database"],
                         "path_model": cf_mlp_riga_embed["save_path"],
                         "watermark_size": watermark_size,

                         # Latent space parameters
                         "mean": 0,
                         "std": 1,
                         "square_size": 5,
                         "start_x": 0,
                         "start_y": 0,
                         "square_value": 1,
                         "n_features": 0.5,
                         "batch_size": int(cf_mlp_riga_dict["batch_size"]*0.5),

                         "epochs": epoch_attack,
                         "epoch_check": epoch_check,

                         # Latent space parameters
                         "lambda": 1,
                         "layer_name": layer_name,
                         "lr": 1e-3,

                         "wd": 0,
                         "scheduler": "MultiStepLR",
                         "architecture": cf_mlp_riga_dict["criterion"],
                         "save_path": "results/attacks/overwriting/diction/" + cf_mlp_riga_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",
                         "momentum": 0,
                         "milestones": [5, 10],
                         "gamma": 0.1,
                         "criterion": cf_mlp_riga_embed["criterion"],
                         "device": cf_mlp_riga_embed["device"],
                         "save_fig_path": "results/attacks/overwriting/diction/" + cf_mlp_riga_embed[
                             "architecture"].lower() + "/" + save_path_attack + ".png",
                         "x_label": "Partition_diction_" + cf_mlp_riga_embed["architecture"].lower(),
                         "y_label": "BER/ACC",
                         "show_acc_epoch": False
                         }

cf_non_watermarked = deepcopy(cf_mlp_riga_dict)
cf_non_watermarked["epochs"] = 2
cf_non_watermarked["show_acc_epoch"] = False
cf_watermarked = deepcopy(cf_mlp_riga_embed)
cf_watermarked["epochs"] = 10
cf_watermarked["show_acc_epoch"] = False
nb_examples = 1224
save_path_attack = "_l" + layer_name + "_wat" + str(
    cf_watermarked["watermark_size"]) + "_ep" + str(cf_non_watermarked["epochs"]) + "_nb_examples" + str(nb_examples)

cf_mlp_riga_attack_pia = {"train_non_watermarked": cf_non_watermarked,
                          "train_watermarked": cf_watermarked,
                          "save_path": "results/attacks/pia/diction/" + cf_mlp_riga_dict[
                              "architecture"].lower() + "/" + save_path_attack + ".pth",
                          "nb_examples": nb_examples
                          }
