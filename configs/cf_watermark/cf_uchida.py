from configs.cf_train.cf_cnn import cf_cnn_dict
from configs.cf_train.cf_mlp import cf_mlp_dict
from configs.cf_train.cf_mlp_riga import cf_mlp_riga_dict

# ************* watermarking MLP architecture ****************
watermark_size = 256
epochs_embed = 1000
layer_name = "fc2.weight"
epoch_check = 30
save_path_embed = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epochs_embed) + \
                  "_epc" + str(epoch_check)

cf_mlp_embed = {"configuration": cf_mlp_dict,
                "database": cf_mlp_dict["database"],
                "watermark_size": watermark_size,
                "epochs": epochs_embed,
                "epoch_check": epoch_check,

                "lambda_1": 0.01,

                "layer_name": layer_name,
                "lr": 1e-3,
                "wd": 1e-4,
                "opt": "Adam",
                "scheduler": "MultiStepLR",
                "architecture": cf_mlp_dict["architecture"],

                "path_model": cf_mlp_dict["save_path"],
                "save_path": "results/watermarked_models/uchida/" + cf_mlp_dict['architecture'].lower() + "/"
                             + save_path_embed + ".pth",

                "momentum": 0,
                "milestones": [100, 150],
                "gamma": 0,
                "criterion": cf_mlp_dict["criterion"],
                "device": cf_mlp_dict["device"],
                }

epoch_attack = 50
lr_attack = 1e-4
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack)

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
                    "save_path": "results/attacks/finetuning/uchida/" + cf_mlp_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",

                    "save_fig_path": "results/attacks/finetuning/uchida/" + cf_mlp_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".png",
                    "x_label": "Partition_uchida_" + cf_mlp_dict["architecture"].lower(),
                    "y_label": "BER/ACC",
                    "show_acc_epoch": False
                    }

cf_mlp_attack_pr = {"configuration": cf_mlp_dict,
                    "path_model": cf_mlp_embed["save_path"],
                    "watermark_size": watermark_size,
                    "criterion": cf_mlp_dict["criterion"],
                    "device": cf_mlp_dict["device"],
                    "architecture": cf_mlp_dict["architecture"],
                    "save_path": "results/attacks/pruning/uchida/" + cf_mlp_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",
                    }

layer_name = "fc2.weight"
epoch_attack = 100
watermark_size = 256
epoch_check = 30
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + \
                   "_epc" + str(epoch_check)

cf_mlp_attack_ow = {"configuration": cf_mlp_dict,
                    "database": cf_mlp_dict["database"],
                    "watermark_size": watermark_size,
                    "epochs": epochs_embed,
                    "epoch_check": epoch_check,

                    "lambda_1": 1,

                    "layer_name": layer_name,
                    "lr": 1e-3,
                    "wd": 1e-4,
                    "opt": "Adam",
                    "scheduler": "MultiStepLR",
                    "architecture": cf_mlp_dict["architecture"],

                    "save_path": "results/watermarked_models/uchida/" + cf_mlp_dict['architecture'].lower() + "/"
                                 + save_path_embed + ".pth",

                    "momentum": 0,
                    "milestones": [1000, 2000],
                    "gamma": 0,
                    "criterion": cf_mlp_dict["criterion"],
                    "device": cf_mlp_dict["device"],
                    "save_fig_path": "results/attacks/overwriting/uchida/" + cf_mlp_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".png",
                    "x_label": "Partition_uchida" + cf_mlp_dict["architecture"].lower(),
                    "y_label": "BER/ACC",
                    "show_acc_epoch": False
                    }
cf_mlp_attack_pia = {}

# ************* watermarking CNN architecture ****************
watermark_size = 256
epochs_embed = 1000
layer_name = "fc1.weight"
epoch_check = 10
save_path_embed = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epochs_embed)

cf_cnn_embed = {"configuration": cf_cnn_dict,
                "database": cf_cnn_dict["database"],
                "watermark_size": watermark_size,
                "epochs": epochs_embed,
                "epoch_check": epoch_check,

                "lambda_1": 1,

                "layer_name": layer_name,

                "lr": 1e-3,
                "wd": 1e-4,
                "opt": "Adam",
                "scheduler": "MultiStepLR",
                "architecture": cf_cnn_dict['architecture'],

                "save_path": "results/watermarked_models/uchida/" + cf_cnn_dict['architecture'].lower() + "/"
                             + save_path_embed + ".pth",

                "path_model": cf_cnn_dict["save_path"],

                "momentum": 0,
                "milestones": [2000, 3000],
                "gamma": .1,
                "criterion": cf_cnn_dict["criterion"],
                "device": cf_cnn_dict["device"],
                }

# ************* watermarking MLP RIGA architecture ****************
watermark_size = 256
epochs_embed = 1000
layer_name = "fc1.weight"
epoch_check = 30
save_path_embed = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epochs_embed) + \
                  "_epc" + str(epoch_check)

cf_mlp_riga_embed = {"configuration": cf_mlp_riga_dict,
                     "database": cf_mlp_riga_dict["database"],
                     "watermark_size": watermark_size,
                     "epochs": epochs_embed,
                     "epoch_check": epoch_check,

                     "lambda_1": 1,

                     "layer_name": layer_name,
                     "lr": 1e-3,
                     "wd": 1e-4,
                     "opt": "Adam",
                     "scheduler": "MultiStepLR",
                     "architecture": cf_mlp_riga_dict["architecture"],

                     "path_model": cf_mlp_riga_dict["save_path"],
                     "save_path": "results/watermarked_models/uchida/" + cf_mlp_riga_dict['architecture'].lower() + "/"
                                  + save_path_embed + ".pth",

                     "momentum": 0,
                     "milestones": [100, 150],
                     "gamma": 0,
                     "criterion": cf_mlp_riga_dict["criterion"],
                     "device": cf_mlp_riga_dict["device"],
                     }

epoch_attack = 50
lr_attack = 1e-4
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack)

cf_mlp_riga_attack_ft = {"configuration": cf_mlp_riga_dict,
                         "database": cf_mlp_riga_dict["database"],
                         "path_model": cf_mlp_riga_embed["save_path"],
                         "watermark_size": watermark_size,
                         "epochs": epoch_attack,
                         "lr": lr_attack,
                         "wd": 1e-5,
                         "opt": "Adam",
                         "scheduler": "MultiStepLR",
                         "architecture": cf_mlp_riga_dict["architecture"],
                         "momentum": cf_mlp_riga_dict["momentum"],
                         "milestones": [150, 200],
                         "gamma": cf_mlp_riga_dict["momentum"],
                         "criterion": cf_mlp_riga_dict["criterion"],

                         "device": cf_mlp_riga_dict["device"],
                         "save_path": "results/attacks/finetuning/uchida/" + cf_mlp_riga_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",

                         "save_fig_path": "results/attacks/finetuning/uchida/" + cf_mlp_riga_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".png",
                         "x_label": "Partition_uchida_" + cf_mlp_riga_dict["architecture"].lower(),
                         "y_label": "BER/ACC",
                         "show_acc_epoch": False
                         }

cf_mlp_riga_attack_pr = {"configuration": cf_mlp_riga_dict,
                         "path_model": cf_mlp_riga_embed["save_path"],
                         "watermark_size": watermark_size,
                         "criterion": cf_mlp_riga_dict["criterion"],
                         "device": cf_mlp_riga_dict["device"],
                         "architecture": cf_mlp_riga_dict["architecture"],
                         "save_path": "results/attacks/pruning/uchida/" + cf_mlp_riga_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",
                         }

layer_name = "fc1.weight"
epoch_attack = 101
watermark_size = 256
epoch_check = 20
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + \
                   "_epc" + str(epoch_check)

cf_mlp_riga_attack_ow = {"configuration": cf_mlp_riga_dict,
                         "database": cf_mlp_riga_dict["database"],
                         "watermark_size": watermark_size,
                         "epochs": epochs_embed,
                         "epoch_check": epoch_check,

                         "lambda_1": 1,

                         "layer_name": layer_name,
                         "lr": 1e-3,
                         "wd": 1e-4,
                         "opt": "Adam",
                         "scheduler": "MultiStepLR",
                         "architecture": cf_mlp_riga_dict["architecture"],

                         "save_path": "results/watermarked_models/uchida/" + cf_mlp_riga_dict[
                             'architecture'].lower() + "/"
                                      + save_path_embed + ".pth",

                         "momentum": 0,
                         "milestones": [1000, 2000],
                         "gamma": 0,
                         "criterion": cf_mlp_riga_dict["criterion"],
                         "device": cf_mlp_riga_dict["device"],
                         "save_fig_path": "results/attacks/overwriting/uchida/" + cf_mlp_riga_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".png",
                         "x_label": "Partition_uchida" + cf_mlp_riga_dict["architecture"].lower(),
                         "y_label": "BER/ACC",
                         "show_acc_epoch": False
                         }

cf_mlp_riga_attack_pia = {}
