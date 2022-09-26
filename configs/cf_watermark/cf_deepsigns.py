from configs.cf_train.cf_mlp import cf_mlp_dict
from configs.cf_train.cf_cnn import cf_cnn_dict
from configs.cf_train.cf_mlp_riga import cf_mlp_riga_dict
from configs.cf_train.cf_resnet18 import cf_resnet18_dict

# ************* watermarking MLP architecture ****************
watermark_size = 128
epochs_embed = 1000
layer_name = "fc2"
nb_wat_classes = 2
epoch_check = 10
save_path_embed = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epochs_embed) + "_nbw" \
                  + str(nb_wat_classes) + "_epc" + str(epoch_check)

cf_mlp_embed = {"configuration": cf_mlp_dict,
                "database": cf_mlp_dict["database"],
                "watermark_size": watermark_size,
                "epochs": epochs_embed,
                "epoch_check": epoch_check,

                "batch_size": int(60000 * 0.01),
                "lambda_1": 0.1,
                "lambda_2": 1,

                "n_features": 512,
                "n_components": 10,
                "nb_wat_classes": nb_wat_classes,  # number of classes to watermark "s" in DeepSigns paper
                "percent_ts": 0.01,

                "layer_name": layer_name,
                "lr": 1e-4,
                "wd": 0,
                "opt": "Adam",
                "scheduler": "MultiStepLR",
                "architecture": cf_mlp_dict["architecture"],

                "path_model": cf_mlp_dict["save_path"],
                "save_path": "results/watermarked_models/deepsigns/" + cf_mlp_dict['architecture'].lower() + "/"
                             + save_path_embed + ".pth",
                "path_gmm": "results/gmm/" + cf_mlp_dict["architecture"].lower() + "/db" + cf_mlp_dict[
                    "database"] + ".pth",  # relu but change in mlp cd

                "momentum": 0,
                "milestones": [100, 150],
                "gamma": 0,
                "criterion": cf_mlp_dict["criterion"],
                "device": cf_mlp_dict["device"],
                }

epoch_attack = 50
lr_attack = 1e-3
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + "_nbw" \
                   + str(nb_wat_classes)

cf_mlp_attack_ft = {"configuration": cf_mlp_dict,
                    "database": cf_mlp_dict["database"],
                    "path_model": cf_mlp_embed["save_path"],
                    "watermark_size": watermark_size,
                    "epochs": epoch_attack,
                    "lr": lr_attack,
                    "wd": cf_mlp_dict["wd"],
                    "opt": "Adam",
                    "scheduler": "MultiStepLR",
                    "architecture": cf_mlp_dict["architecture"],
                    "momentum": cf_mlp_dict["momentum"],
                    "milestones": [150, 200],
                    "gamma": cf_mlp_dict["momentum"],
                    "criterion": cf_mlp_dict["criterion"],

                    "device": cf_mlp_dict["device"],
                    "save_path": "results/attacks/finetuning/deepsigns/" + cf_mlp_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",

                    "save_fig_path": "results/attacks/finetuning/deepsigns/" + cf_mlp_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".png",
                    "x_label": "Partition_deepsign_" + cf_mlp_dict["architecture"].lower(),
                    "y_label": "BER/ACC",
                    "show_acc_epoch": False
                    }

cf_mlp_attack_pr = {"configuration": cf_mlp_dict,
                    "path_model": cf_mlp_embed["save_path"],
                    "watermark_size": watermark_size,
                    "amount": 0.7,
                    "criterion": cf_mlp_dict["criterion"],
                    "device": cf_mlp_dict["device"],
                    "architecture": cf_mlp_dict["architecture"],
                    "save_path": "results/attacks/pruning/deepsigns/" + cf_mlp_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",
                    }

layer_name = "fc2"
epoch_attack = 100
watermark_size = 128
nb_wat_classes = 2
epoch_check = 10
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + "_nbw" \
                   + str(nb_wat_classes)

cf_mlp_attack_ow = {"configuration": cf_mlp_dict,
                    "database": cf_mlp_dict["database"],
                    "watermark_size": watermark_size,
                    "epochs": epoch_attack,
                    "epoch_check": epoch_check,
                    "batch_size": int(60000 * 0.01),
                    "lambda_1": 1,
                    "lambda_2": 1,

                    "n_features": 512,
                    "n_components": 10,
                    "nb_wat_classes": nb_wat_classes,  # number of classes to watermark "s" in DeepSigns paper
                    "percent_ts": 0.01,

                    "layer_name": layer_name,
                    "lr": 1e-4,
                    "wd": 0,
                    "opt": "Adam",
                    "scheduler": "MultiStepLR",
                    "architecture": cf_mlp_dict["architecture"],

                    "path_model": cf_mlp_embed["save_path"],
                    "path_gmm": "results/gmm/" + cf_mlp_dict["architecture"].lower() + "/db" + cf_mlp_dict[
                        "database"] + "_ow.pth",
                    "save_path": "results/attacks/overwriting/deepsigns/" + cf_mlp_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",

                    "momentum": 0,
                    "milestones": [1000, 2000],
                    "gamma": 0,
                    "criterion": cf_mlp_dict["criterion"],
                    "device": cf_mlp_dict["device"],
                    "save_fig_path": "results/attacks/overwriting/deepsigns/" + cf_mlp_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".png",
                    "x_label": "Partition_deepsigns" + cf_mlp_dict["architecture"].lower(),
                    "y_label": "BER/ACC",
                    "show_acc_epoch": False
                    }

cf_mlp_attack_pia = {}

# ************* watermarking CNN architecture ****************
watermark_size = 64
epochs_embed = 1000
layer_name = "fc1"
nb_wat_classes = 4
epoch_check = 20
save_path_embed = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epochs_embed) + "_nbw" \
                  + str(nb_wat_classes)
cf_cnn_embed = {"configuration": cf_cnn_dict,
                "database": cf_cnn_dict["database"],
                "watermark_size": watermark_size,
                "epochs": epochs_embed,
                "epoch_check": epoch_check,
                "batch_size": int(50000 * 0.01),

                "lambda_1": 1,
                "lambda_2": 1,
                "n_features": 512,
                "n_components": 10,
                "nb_wat_classes": nb_wat_classes,  # number of classes to watermark
                "percent_ts": 0.01,
                "layer_name": layer_name,

                "lr": 1e-4,
                "wd": 0,
                "opt": "Adam",
                "scheduler": "MultiStepLR",
                "architecture": cf_cnn_dict['architecture'],

                "save_path": "results/watermarked_models/deepsigns/" + cf_cnn_dict['architecture'].lower() + "/"
                             + save_path_embed + ".pth",
                "path_gmm": "results/gmm/" + cf_cnn_dict["architecture"].lower() + "/db" + cf_cnn_dict[
                    "database"] + ".pth",  # relu but change in mlp cd
                "path_model": cf_cnn_dict["save_path"],

                "momentum": 0,
                "milestones": [2000, 3000],
                "gamma": .1,
                "criterion": cf_cnn_dict["criterion"],
                "device": cf_cnn_dict["device"],
                }

epoch_attack = 10
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + "_nbw" \
                   + str(nb_wat_classes)

cf_cnn_attack_ft = {"configuration": cf_cnn_dict,
                    "database": cf_cnn_dict["database"],
                    "path_model": cf_cnn_embed['save_path'],
                    "epochs": epoch_attack,
                    "watermark_size": watermark_size,
                    "batch_size": cf_cnn_dict["batch_size"],
                    "layer_name": layer_name,

                    "lr": 1e-4,
                    "wd": 0,

                    "scheduler": "MultiStepLR",
                    "opt": "Adam",
                    "architecture": cf_cnn_dict["architecture"],
                    "save_path": "results/attacks/finetuning/deepsigns/" + cf_cnn_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",
                    "momentum": 0.9,
                    "milestones": [150, 200],
                    "gamma": 0.1,
                    "criterion": cf_cnn_dict["criterion"],
                    "device": cf_cnn_dict["device"],
                    "save_fig_path": "results/attacks/finetuning/deepsigns/" + cf_cnn_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".png",
                    "x_label": "Partition_deepsign_CNN",
                    "y_label": "BER/ACC",
                    "show_acc_epoch": False
                    }

cf_cnn_attack_pr = {"configuration": cf_cnn_dict,
                    "database": cf_cnn_dict["database"],
                    "path_model": cf_cnn_embed["save_path"],
                    "watermark_size": watermark_size,
                    "amount": 0.1,
                    "architecture": cf_cnn_dict['architecture'],
                    "save_path": "results/attacks/pruning/deepsigns/" + cf_cnn_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",
                    "criterion": cf_cnn_dict["criterion"],
                    "device": cf_cnn_dict["device"],

                    }

layer_name = "fc1"
epoch_attack = 100
watermark_size = 128
nb_wat_classes = 2
epoch_check = 20
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + "_nbw" \
                   + str(nb_wat_classes)
cf_cnn_attack_ow = {"configuration": cf_cnn_dict,
                    "database": cf_cnn_dict["database"],
                    "watermark_size": watermark_size,
                    "epochs": epochs_embed,
                    "epoch_check": epoch_check,
                    "batch_size": int(50000 * 0.01),
                    "lambda_1": 0.1,
                    "lambda_2": 0.1,

                    "n_features": 512,
                    "n_components": 10,
                    "nb_wat_classes": nb_wat_classes,  # number of classes to watermark "s" in DeepSigns paper
                    "percent_ts": 0.01,

                    "layer_name": layer_name,
                    "lr": 1e-4,
                    "wd": 0,
                    "opt": "Adam",
                    "scheduler": "MultiStepLR",
                    "architecture": cf_cnn_dict["architecture"],

                    "path_model": cf_cnn_embed["save_path"],
                    "path_gmm": "results/gmm/" + cf_cnn_dict["architecture"].lower() + "/db" + cf_cnn_dict[
                        "database"] + "_ow.pth",
                    "save_path": "results/attacks/overwriting/deepsigns/" + cf_cnn_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",

                    "momentum": 0,
                    "milestones": [1000, 2000],
                    "gamma": 0,
                    "criterion": cf_cnn_dict["criterion"],
                    "device": cf_cnn_dict["device"],
                    "save_fig_path": "results/attacks/overwriting/deepsigns/" + cf_cnn_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".png",
                    "x_label": "Partition_deepsign_c" + cf_cnn_dict["architecture"].lower(),
                    "y_label": "BER/ACC",
                    "show_acc_epoch": False
                    }

cf_cnn_attack_pia = {}
# ************* watermarking Resnet18 architecture ****************
watermark_size = 128
epochs_embed = 1000
layer_name = "view"
nb_wat_classes = 2
epoch_check = 20
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + "_nbw" \
                   + str(nb_wat_classes)

cf_resnet18_embed = {"configuration": cf_resnet18_dict,
                     "database": cf_resnet18_dict["database"],
                     "watermark_size": watermark_size,
                     "epochs": epochs_embed,
                     "epoch_check": epoch_check,
                     "batch_size": int(50000 * 0.01),

                     "lambda_1": 1,
                     "lambda_2": 1,
                     "n_features": 512,
                     "n_components": 10,
                     "nb_wat_classes": nb_wat_classes,  # number of classes to watermark
                     "percent_ts": 0.01,
                     "layer_name": layer_name,
                     "lr": 1e-4,
                     "wd": 0,
                     "opt": "Adam",
                     "scheduler": "MultiStepLR",
                     "architecture": cf_resnet18_dict['architecture'],

                     "save_path": "results/watermarked_models/deepsigns/" + cf_resnet18_dict[
                         'architecture'].lower() + "/" + save_path_embed + ".pth",
                     "path_gmm": "results/gmm/" + cf_resnet18_dict["architecture"].lower() + "/_db" +
                                 cf_resnet18_dict[
                                     "database"] + ".pth",
                     # relu but change in mlp cd
                     "path_model": cf_resnet18_dict["save_path"],
                     "momentum": 0,
                     "milestones": [100, 150],
                     "gamma": 0,
                     "criterion": cf_resnet18_dict["criterion"],
                     "device": cf_resnet18_dict["device"],
                     }

epoch_attack = 50
save_path_attack = layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + "_nbw" \
                   + str(nb_wat_classes)

cf_resnet18_attack_ft = {"configuration": cf_resnet18_dict,
                         "database": cf_resnet18_dict["database"],
                         "path_model": cf_resnet18_embed['save_path'],
                         "epochs": epoch_attack,
                         "watermark_size": watermark_size,
                         "batch_size": 512,

                         "layer_name": layer_name,
                         "lr": 1e-4,
                         "wd": 0,
                         "scheduler": "MultiStepLR",
                         "opt": "Adam",
                         "architecture": cf_resnet18_dict["architecture"],
                         "save_path": "results/attacks/finetuning/deepsigns/" + cf_resnet18_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",
                         "momentum": 0,
                         "milestones": [150, 200],
                         "gamma": 0,
                         "criterion": cf_resnet18_dict["criterion"],
                         "device": cf_resnet18_dict["device"],
                         "save_fig_path": "results/attacks/finetuning/deepsigns/" + cf_resnet18_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".png",
                         "x_label": "Partition_deepsigns_resnet18",
                         "y_label": "BER/ACC",
                         "show_acc_epoch": False
                         }

cf_resnet18_attack_pr = {"configuration": cf_resnet18_dict,
                         "database": cf_resnet18_dict["database"],
                         "path_model": cf_resnet18_embed["save_path"],
                         "watermark_size": watermark_size,
                         "criterion": cf_resnet18_dict["criterion"],
                         "device": cf_resnet18_dict["device"],
                         "amount": 0.8,
                         "architecture": cf_resnet18_dict["architecture"],
                         "save_path": "results/attacks/pruning/deepsigns/" + cf_resnet18_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",

                         }
watermark_size = 128
layer_name = "view"
nb_wat_classes = 4
epoch_attack = 100
epoch_check = 20
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + "_nbw" \
                   + str(nb_wat_classes)

cf_resnet18_attack_ow = {"configuration": cf_resnet18_dict,
                         "database": cf_resnet18_dict["database"],
                         "path_model": cf_resnet18_embed["save_path"],
                         "watermark_size": watermark_size,
                         "batch_size": int(50000 * 0.01),
                         "lambda_1": 0.1,
                         "lambda_2": 0.1,

                         "n_features": 512,
                         "n_components": 10,
                         "nb_wat_classes": nb_wat_classes,  # number of classes to watermark "s" in DeepSigns paper
                         "percent_ts": 0.01,
                         "epochs": epoch_attack,
                         "epoch_check": epoch_check,
                         "layer_name": layer_name,
                         "lr": 1e-4,
                         "wd": 0,
                         "opt": "Adam",
                         "scheduler": "MultiStepLR",
                         "architecture": cf_resnet18_dict["architecture"],
                         "save_path": "results/attacks/overwriting/deepsigns/" + cf_resnet18_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",
                         "path_gmm": "results/gmm/" + cf_resnet18_dict["architecture"].lower() + "/_db" +
                                     cf_resnet18_dict[
                                         "database"] + "_ow.pth",
                         "momentum": 0,
                         "milestones": [1000, 2000],
                         "gamma": 0,
                         "criterion": cf_resnet18_dict["criterion"],
                         "device": cf_resnet18_dict["device"],
                         "save_fig_path": "results/attacks/overwriting/deepsigns/" + cf_resnet18_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".png",
                         "x_label": "Partition_deepsigns" + cf_resnet18_dict["architecture"].lower(),
                         "y_label": "BER/ACC",
                         "show_acc_epoch": False
                         }

cf_resnet18_attack_pia = {}

# ************* watermarking MLP_RIGA architecture ****************
watermark_size = 128
epochs_embed = 10000
layer_name = "fc2"
nb_wat_classes = 2
epoch_check = 30
save_path_embed = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epochs_embed) + "_nbw" \
                  + str(nb_wat_classes) + "_epc" + str(epoch_check)

cf_mlp_riga_embed = {"configuration": cf_mlp_riga_dict,
                     "database": cf_mlp_riga_dict["database"],
                     "watermark_size": watermark_size,
                     "epochs": epochs_embed,
                     "epoch_check": epoch_check,

                     "batch_size": int(60000 * 0.01),
                     "lambda_1": 1,
                     "lambda_2": 1,

                     "n_features": 64,
                     "n_components": 10,
                     "nb_wat_classes": nb_wat_classes,  # number of classes to watermark "s" in DeepSigns paper
                     "percent_ts": 0.01,

                     "layer_name": layer_name,
                     "lr": 1e-3,
                     "wd": 0,
                     "opt": "Adam",
                     "scheduler": "MultiStepLR",
                     "architecture": cf_mlp_riga_dict["architecture"],

                     "path_model": cf_mlp_riga_dict["save_path"],
                     "save_path": "results/watermarked_models/deepsigns/" + cf_mlp_riga_dict[
                         'architecture'].lower() + "/"
                                  + save_path_embed + ".pth",
                     "path_gmm": "results/gmm/" + cf_mlp_riga_dict["architecture"].lower() + "/_db" + cf_mlp_riga_dict[
                         "database"] + ".pth",  # relu but change in mlp cd

                     "momentum": 0,
                     "milestones": [100, 150],
                     "gamma": 0.1,
                     "criterion": cf_mlp_riga_dict["criterion"],
                     "device": cf_mlp_riga_dict["device"],
                     }

epoch_attack = 50
lr_attack = 1e-4
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + "_nbw" \
                   + str(nb_wat_classes)

cf_mlp_riga_attack_ft = {"configuration": cf_mlp_riga_embed,
                         "database": cf_mlp_riga_embed["database"],
                         "path_model": cf_mlp_riga_embed["save_path"],
                         "watermark_size": watermark_size,
                         "epochs": epoch_attack,
                         "lr": lr_attack,
                         "wd": cf_mlp_riga_dict["wd"],
                         "opt": "Adam",
                         "scheduler": "MultiStepLR",
                         "architecture": cf_mlp_riga_dict["architecture"],
                         "momentum": cf_mlp_riga_dict["momentum"],
                         "milestones": [150, 200],
                         "gamma": cf_mlp_riga_dict["momentum"],
                         "criterion": cf_mlp_riga_dict["criterion"],

                         "device": cf_mlp_riga_dict["device"],
                         "save_path": "results/attacks/finetuning/deepsigns/" + cf_mlp_riga_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",

                         "save_fig_path": "results/attacks/finetuning/deepsigns/" + cf_mlp_riga_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".png",
                         "x_label": "Partition_deepsign_" + cf_mlp_riga_dict["architecture"].lower(),
                         "y_label": "BER/ACC",
                         "show_acc_epoch": False
                         }

cf_mlp_riga_attack_pr = {"configuration": cf_mlp_riga_dict,
                         "path_model": cf_mlp_riga_embed["save_path"],
                         "watermark_size": watermark_size,
                         "amount": 0.7,
                         "criterion": cf_mlp_riga_dict["criterion"],
                         "device": cf_mlp_riga_dict["device"],
                         "architecture": cf_mlp_dict["architecture"],
                         "save_path": "results/attacks/pruning/deepsigns/" + cf_mlp_riga_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",
                         }

layer_name = "fc2"
epoch_attack = 100
watermark_size = 256
nb_wat_classes = 2
epoch_check = 30
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + "_nbw" \
                   + str(nb_wat_classes)

cf_mlp_riga_attack_ow = {"configuration": cf_mlp_riga_dict,
                         "database": cf_mlp_riga_dict["database"],
                         "watermark_size": watermark_size,
                         "epochs": epoch_attack,
                         "epoch_check": epoch_check,
                         "batch_size": int(60000 * 0.01),
                         "lambda_1": 1,
                         "lambda_2": 1,

                         "n_features": 64,
                         "n_components": 10,
                         "nb_wat_classes": nb_wat_classes,  # number of classes to watermark "s" in DeepSigns paper
                         "percent_ts": 0.01,

                         "layer_name": layer_name,
                         "lr": 1e-4,
                         "wd": 0,
                         "opt": "Adam",
                         "scheduler": "MultiStepLR",
                         "architecture": cf_mlp_riga_dict["architecture"],

                         "path_model": cf_mlp_riga_embed["save_path"],
                         "path_gmm": "results/gmm/" + cf_mlp_riga_dict["architecture"].lower() + "/db" +
                                     cf_mlp_riga_dict[
                                         "database"] + "_ow.pth",
                         "save_path": "results/attacks/overwriting/deepsigns/" + cf_mlp_riga_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",

                         "momentum": 0,
                         "milestones": [1000, 2000],
                         "gamma": 0,
                         "criterion": cf_mlp_riga_dict["criterion"],
                         "device": cf_mlp_riga_dict["device"],
                         "save_fig_path": "results/attacks/overwriting/deepsigns/" + cf_mlp_riga_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".png",
                         "x_label": "Partition_deepsigns" + cf_mlp_riga_dict["architecture"].lower(),
                         "y_label": "BER/ACC",
                         "show_acc_epoch": False
                         }

cf_mlp_riga_attack_pia = {}
