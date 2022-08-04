from configs.cf_train.cf_mlp import cf_mlp_dict_as
from configs.cf_train.cf_cnn import cf_cnn_dict_as
from configs.cf_train.cf_resnet18 import cf_resnet18_dict_as

from copy import deepcopy

# ************* watermarking MLP architecture ****************
watermark_size = 256
epochs_embed = 10000
layer_name = "fc2"
save_path_embed = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epochs_embed)

cf_mlp_embed = {"configuration": cf_mlp_dict_as,
                "database": cf_mlp_dict_as["database"],
                "watermark_size": watermark_size,
                "epochs": epochs_embed,

                # Latent space parameters
                "batch_size": 128,
                "mean": 0,
                "std": 1,
                "n_features": 512 // 32,
                "lambda_1": 1e-0,

                "layer_name": layer_name,
                "lr": 1e-4,
                "wd": 0,
                "opt": "Adam",
                "scheduler": "MultiStepLR",
                "architecture": cf_mlp_dict_as["architecture"],

                "path_model": cf_mlp_dict_as["save_path"],
                "save_path": "results/watermarked_models/diction/" + cf_mlp_dict_as['architecture'].lower() + "/"
                             + save_path_embed + ".pth",
                "momentum": 0,
                "milestones": [100, 1000],
                "gamma": 0.1,
                "criterion": cf_mlp_dict_as["criterion"],
                "device": cf_mlp_dict_as["device"]
                }

epoch_attack = 12
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + ".pth"

cf_mlp_attack_ft = {"configuration": cf_mlp_dict_as,
                    "database": cf_mlp_dict_as["database"],
                    "watermark_size": watermark_size,
                    "epochs": epoch_attack,
                    "lr": 1e-3,
                    "wd": 0,
                    "opt": "Adam",
                    "scheduler": "MultiStepLR",
                    "architecture": cf_mlp_dict_as["architecture"],

                    "path_model": cf_mlp_embed["save_path"],
                    "save_path": "results/attacks/finetuning/diction/" + cf_mlp_dict_as[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",
                    "momentum": 0,
                    "milestones": [100, 2000],
                    "gamma": 0,
                    "criterion": cf_mlp_dict_as["criterion"],
                    "device": cf_mlp_dict_as["device"],
                    "save_fig_path": "results/attacks/finetuning/diction/" + cf_mlp_dict_as[
                        "architecture"].lower() + "/" + save_path_attack + ".png",
                    "x_label": "Partition_diction_" + cf_mlp_dict_as["architecture"].lower(),
                    "y_label": "BER/ACC",
                    "show_acc_epoch": False
                    }

cf_mlp_attack_pr = {"configuration": cf_mlp_dict_as,
                    "database": cf_mlp_dict_as["database"],
                    "path_model": cf_mlp_embed["save_path"],
                    "watermark_size": watermark_size,
                    "amount": 0.9,
                    "save_path": "results/attacks/pruning/diction/" + cf_mlp_dict_as[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",
                    "architecture": cf_mlp_dict_as["architecture"],
                    }

epoch_attack = 60
watermark_size = 512
save_path_attack = "_b" + "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack)
cf_mlp_attack_ow = {"configuration": cf_cnn_dict_as,
                    "database": cf_mlp_dict_as["database"],
                    "path_model": cf_mlp_embed["save_path"],
                    "watermark_size": watermark_size,

                    # Latent space parameters
                    "mean": 0.26,
                    "std": 0.1,
                    "n_features": 512,
                    "batch_size": 128,

                    "epochs": epoch_attack,
                    "layer_name": layer_name,
                    "lr": 1e-3,
                    "wd": 0,
                    "scheduler": "MultiStepLR",
                    "architecture": cf_mlp_dict_as["criterion"],
                    "save_path": "results/attacks/overwriting/diction/" + cf_mlp_dict_as[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",
                    "momentum": 0,
                    "milestones": [5, 10],
                    "gamma": 0.1,
                    "criterion": cf_mlp_dict_as["criterion"],
                    "device": cf_mlp_dict_as["device"],
                    "save_fig_path": "results/attacks/overwriting/diction/" + cf_mlp_dict_as[
                        "architecture"].lower() + "/" + save_path_attack + ".png",
                    "attack_set_size": cf_mlp_dict_as["attack_set_size"],
                    "x_label": "Partition_diction_" + cf_mlp_dict_as["architecture"].lower(),
                    "y_label": "BER/ACC",
                    "show_acc_epoch": False
                    }

cf_non_watermarked = deepcopy(cf_mlp_dict_as)
cf_non_watermarked["epochs"] = 2
cf_non_watermarked["show_acc_epoch"] = False
cf_watermarked = deepcopy(cf_mlp_embed)
cf_watermarked["epochs"] = 10
cf_watermarked["show_acc_epoch"] = False
nb_examples = 1224
save_path_attack = "_l" + layer_name + "_wat" + str(
    cf_watermarked["watermark_size"]) + "_ep" + str(cf_non_watermarked["epochs"]) + "_nb_examples" + str(nb_examples)

cf_mlp_attack_pia = {"train_non_watermarked": cf_non_watermarked,
                     "train_watermarked": cf_watermarked,
                     "save_path": "results/attacks/pia/diction/" + cf_mlp_dict_as[
                         "architecture"].lower() + "/" + save_path_attack + ".pth",
                     "nb_examples": nb_examples
                     }

# ************* watermarking CNN architecture ****************
watermark_size = 256
epochs_embed = 10000
layer_name = "fc1"
save_path_embed = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epochs_embed)
cf_cnn_embed = {"configuration": cf_cnn_dict_as,
                "database": cf_cnn_dict_as["database"],
                "watermark_size": watermark_size,
                "epochs": epochs_embed,

                # Latent space parameters
                "batch_size": 10,
                "mean": 0,
                "std": 1,
                "lambda_1": 1e-0,
                "n_features": 512 // 32,
                "layer_name": layer_name,

                "lr": 1e-4,
                "wd": 0,
                "opt": "Adam",
                "scheduler": "MultiStepLR",
                "architecture": cf_cnn_dict_as['architecture'],
                "momentum": 0,
                "milestones": [300, 3000],
                "gamma": .1,
                "criterion": cf_cnn_dict_as["criterion"],
                "device": cf_cnn_dict_as["device"],

                "path_model": cf_cnn_dict_as["save_path"],
                "save_path": "results/watermarked_models/diction/" + cf_cnn_dict_as['architecture'].lower() + "/"
                             + save_path_embed + ".pth",
                }

epoch_attack = 10
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack)

cf_cnn_attack_ft = {"configuration": cf_cnn_dict_as,
                    "database": cf_cnn_dict_as["database"],
                    "path_model": cf_cnn_embed['save_path'],
                    "epochs": epoch_attack,
                    "watermark_size": watermark_size,

                    "lr": 1e-4,
                    "wd": 0,
                    "scheduler": "MultiStepLR",
                    "opt": "Adam",
                    "architecture": cf_cnn_dict_as["architecture"],

                    "momentum": 0,
                    "milestones": [150, 200],
                    "gamma": 0.1,
                    "criterion": cf_cnn_dict_as["criterion"],
                    "device": cf_cnn_dict_as["device"],

                    "save_path": "results/attacks/finetuning/diction/" + cf_cnn_dict_as[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",
                    "save_fig_path": "results/attacks/finetuning/diction/" + cf_cnn_dict_as[
                        "architecture"].lower() + "/" + save_path_attack + ".png",

                    "x_label": "Partition_diction_CNN",
                    "y_label": "BER/ACC",
                    "show_acc_epoch": False
                    }

cf_cnn_attack_pr = {"configuration": cf_cnn_dict_as,
                    "database": cf_cnn_dict_as["database"],
                    "path_model": cf_cnn_embed["save_path"],
                    "watermark_size": watermark_size,
                    "amount": 0.8,

                    "architecture": cf_cnn_dict_as["architecture"],
                    "save_path": "results/attacks/pruning/diction/" + cf_cnn_dict_as[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",
                    }

layer_name = "fc1"
epoch_attack = 100
watermark_size = 512
save_path_attack = "_b" + "_l" + layer_name + "_wat" + str(
    watermark_size) + "_ep" + str(epoch_attack) + "_pt"
cf_cnn_attack_ow = {"configuration": cf_cnn_dict_as,
                    "database": cf_cnn_dict_as["database"],
                    "path_model": cf_cnn_embed["save_path"],
                    "watermark_size": watermark_size,

                    # Latent space
                    "mean": 0.5,
                    "std": 0.2,
                    "epochs": epoch_attack,
                    "n_features": 512,
                    "batch_size": 128,

                    "layer_name": layer_name,
                    "lr": 1e-4,
                    "wd": 0,
                    "opt": "Adam",
                    "scheduler": "MultiStepLR",
                    "architecture": cf_cnn_dict_as["criterion"],

                    "momentum": 0,
                    "milestones": [1000, 2000],
                    "gamma": 0,
                    "criterion": cf_cnn_dict_as["criterion"],
                    "device": cf_cnn_dict_as["device"],

                    "save_path": "results/attacks/overwriting/diction/" + cf_cnn_dict_as[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",
                    "save_fig_path": "results/attacks/overwriting/diction/" + cf_cnn_dict_as[
                        "architecture"].lower() + "/" + save_path_attack + ".png",
                    "x_label": "Partition_diction_" + cf_cnn_dict_as["architecture"].lower(),
                    "y_label": "BER/ACC",
                    "show_acc_epoch": False
                    }

cf_cnn_attack_pia = {"train_non_watermarked": cf_non_watermarked,
                     "train_watermarked": cf_watermarked,
                     "save_path": "results/attacks/pia/diction/" + cf_cnn_dict_as[
                         "architecture"].lower() + "/" + save_path_attack + ".pth",
                     "nb_examples": nb_examples
                     }
# ************* watermarking Resnet18 architecture ****************
watermark_size = 256
epochs_embed = 1000
layer_name = "view"
save_path_embed = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epochs_embed)

cf_resnet18_embed = {"configuration": cf_resnet18_dict_as,
                     "database": cf_resnet18_dict_as["database"],
                     "watermark_size": watermark_size,
                     "epochs": epochs_embed,

                     # Latent space
                     "mean": 0,
                     "std": 1,
                     "batch_size": 10,
                     "lambda_1": 1e-0,
                     "n_features": 512 // 32,
                     "layer_name": layer_name,

                     "lr": 1e-4,
                     "wd": 0,
                     "opt": "Adam",
                     "scheduler": "MultiStepLR",
                     "architecture": cf_resnet18_dict_as['architecture'],
                     "save_path": "results/watermarked_models/diction/" + cf_resnet18_dict_as[
                         'architecture'].lower() + "/" + save_path_embed + ".pth",
                     # relu but change in mlp cd
                     "path_model": cf_resnet18_dict_as["save_path"],
                     "momentum": 0,
                     "milestones": [50, 150],
                     "gamma": .1,
                     "criterion": cf_resnet18_dict_as["criterion"],
                     "device": cf_resnet18_dict_as["device"]
                     }

epoch_attack = 10
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack)

cf_resnet18_attack_ft = {"configuration": cf_resnet18_dict_as,
                         "database": cf_resnet18_dict_as["database"],
                         "path_model": cf_resnet18_embed['save_path'] + '.pth',
                         "epochs": epoch_attack,
                         "watermark_size": watermark_size,

                         "layer_name": layer_name,
                         "lr": 1e-4,
                         "wd": 0,
                         "scheduler": "MultiStepLR",
                         "opt": "Adam",
                         "architecture": cf_resnet18_dict_as["architecture"],
                         "save_path": "results/attacks/finetuning/diction/" + cf_resnet18_dict_as[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",
                         "momentum": 0,
                         "milestones": [150, 200],
                         "gamma": 0,
                         "criterion": cf_resnet18_dict_as["criterion"],
                         "device": cf_resnet18_dict_as["device"],
                         "save_fig_path": "results/attacks/finetuning/diction/" + cf_resnet18_dict_as[
                             "architecture"].lower() + "/" + save_path_attack + ".png",
                         "x_label": "Partition_diction_resnet18",
                         "y_label": "BER/ACC",
                         "show_acc_epoch": False
                         }

cf_resnet18_attack_pr = {"configuration": cf_resnet18_dict_as,
                         "database": cf_resnet18_dict_as["database"],
                         "path_model": cf_resnet18_embed["save_path"] + ".pth",
                         "watermark_size": watermark_size,
                         "amount": 0.8,
                         "architecture": cf_resnet18_dict_as["architecture"],
                         "save_path": "results/attacks/pruning/diction/" + cf_resnet18_dict_as[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",
                         }

layer_name = "view"
epoch_attack = 100
watermark_size = 512
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack)

cf_resnet18_attack_ow = {"configuration": cf_resnet18_dict_as,
                         "database": cf_resnet18_dict_as["database"],
                         "path_model": cf_resnet18_embed["save_path"] + ".pth",
                         "watermark_size": watermark_size,
                         # Latent space
                         "mean": 0,
                         "std": 1,
                         "n_features": 512,
                         "batch_size": 10,

                         "epochs": epoch_attack,
                         "layer_name": layer_name,
                         "lr": 1e-4,
                         "wd": 0,
                         "opt": "Adam",
                         "scheduler": "MultiStepLR",
                         "architecture": cf_resnet18_dict_as["criterion"],
                         "save_path": "results/attacks/overwriting/diction/" + cf_resnet18_dict_as[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",
                         "momentum": 0,
                         "milestones": [1000, 2000],
                         "gamma": 0,
                         "criterion": cf_resnet18_dict_as["criterion"],
                         "device": cf_resnet18_dict_as["device"],
                         "save_fig_path": "results/attacks/overwriting/diction/" + cf_resnet18_dict_as[
                             "architecture"].lower() + "/" + save_path_attack + ".png",
                         "x_label": "Partition_diction_" + cf_resnet18_dict_as["architecture"].lower(),
                         "y_label": "BER/ACC",
                         "show_acc_epoch": False
                         }

cf_resnet18_attack_pia = {"train_non_watermarked": cf_non_watermarked,
                          "train_watermarked": cf_watermarked,
                          "save_path": "results/attacks/pia/diction/" + cf_resnet18_dict_as[
                              "architecture"].lower() + "/" + save_path_attack + ".pth",
                          "nb_examples": nb_examples
                          }
