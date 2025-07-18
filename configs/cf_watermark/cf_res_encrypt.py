from configs.cf_train.cf_cnn import cf_cnn_dict
from configs.cf_train.cf_mlp import cf_mlp_dict
from configs.cf_train.cf_mlp_riga import cf_mlp_riga_dict
from configs.cf_train.cf_resnet18 import cf_resnet18_dict

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

                "lr_MN": 1e-3,
                "wd_MN": 1e-4,
                "lambda_1": 1.,
                "lambda_2": 1.,
                "lambda_3": 1e-5,
                "expansion_factor": 1,

                "layer_name": layer_name,
                "lr": 1e-3,
                "wd": 1e-4,
                "opt": "Adam",
                "scheduler": "MultiStepLR",
                "architecture": cf_mlp_dict["architecture"],

                "path_model": cf_mlp_dict["save_path"],
                "save_path": "results/watermarked_models/res_encrypt/" + cf_mlp_dict['architecture'].lower() + "/"
                             + save_path_embed + ".pth",

                "momentum": 0,
                "milestones": [30, 150],
                "gamma": .1,
                "criterion": cf_mlp_dict["criterion"],
                "device": cf_mlp_dict["device"],
                }

### ****************************Fine tuning attack***************************************
epoch_attack = 50
lr_attack = 1e-4
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack)
cf_mlp_attack_ft = {"configuration": cf_mlp_dict,
                    "database": cf_mlp_dict["database"],
                    "path_model": cf_mlp_embed["save_path"],
                    "watermark_size": watermark_size,
                    "epochs": epoch_attack,
                    "lr": lr_attack,
                    "wd": 0,
                    "opt": "Adam",
                    "scheduler": "MultiStepLR",
                    "architecture": cf_mlp_dict["architecture"],
                    "momentum": cf_mlp_dict["momentum"],
                    "milestones": [150, 200],
                    "gamma": cf_mlp_dict["momentum"],
                    "criterion": cf_mlp_dict["criterion"],

                    "device": cf_mlp_dict["device"],
                    "save_path": "results/attacks/finetuning/res_encrypt/" + cf_mlp_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",

                    "save_fig_path": "results/attacks/finetuning/res_encrypt/" + cf_mlp_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".png",
                    "x_label": "Partition_res_encrypt_" + cf_mlp_dict["architecture"].lower(),
                    "y_label": "BER/ACC",
                    "show_acc_epoch": False
                    }

### ****************************Pruning attack attack***************************************
cf_mlp_attack_pr = {"configuration": cf_mlp_dict,
                    "path_model": cf_mlp_embed["save_path"],
                    "watermark_size": watermark_size,
                    "criterion": cf_mlp_dict["criterion"],
                    "device": cf_mlp_dict["device"],
                    "architecture": cf_mlp_dict["architecture"],
                    "save_path": "results/attacks/pruning/res_encrypt/" + cf_mlp_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",
                    }

### ****************************Overwriting attack***************************************

layer_name = "fc2.weight"
epoch_attack = 100
watermark_size = 256
epoch_check = 10
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + \
                   "_epc" + str(epoch_check)
cf_mlp_attack_ow = {"configuration": cf_mlp_dict,
                    "database": cf_mlp_dict["database"],
                    "watermark_size": watermark_size,
                    "epochs": epoch_attack,
                    "epoch_check": epoch_check,

                    "lr_MN": 1e-3,
                    "wd_MN": 1e-4,
                    "lambda_1": 1,
                    "lambda_2": 1,
                    "lambda_3": 1e-5,
                    "expansion_factor": 1,

                    "layer_name": layer_name,
                    "lr": 1e-3,
                    "wd": 1e-4,
                    "opt": "Adam",
                    "scheduler": "MultiStepLR",
                    "architecture": cf_mlp_dict["architecture"],

                    "path_model": cf_mlp_embed["save_path"],
                    "save_path": "results/attacks/overwriting/res_encrypt/" + cf_mlp_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",

                    "momentum": 0,
                    "milestones": [10, 2000],
                    "gamma": .1,
                    "criterion": cf_mlp_dict["criterion"],
                    "device": cf_mlp_dict["device"],
                    "save_fig_path": "results/attacks/overwriting/res_encrypt/" + cf_mlp_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".png",
                    "x_label": "Partition_res_encrypt" + cf_mlp_dict["architecture"].lower(),
                    "y_label": "BER/ACC",
                    "show_acc_epoch": False
                    }

### ****************************PIA attack***************************************

cf_mlp_attack_pia = {}

### ****************************Dummy neurons attack***************************************
layer_name="fc2"
num_dummy=2
attack_type="neuron_clique"
save_path_attack = "l_" + layer_name + "_attack_type_" + str(attack_type)

cf_mlp_attack_dummy_neurons = {
    "configuration": cf_mlp_dict,
    "database": cf_mlp_dict["database"],
    "path_model": cf_mlp_embed["save_path"],
    "save_path": "results/attacks/dummy_neurons/res_encrypt/" + cf_mlp_dict[
        "architecture"].lower() + "/" + save_path_attack + ".pth",
    "device":device,
    "attack_type":attack_type,
    "layer_name": layer_name,
    "num_dummy": 2, # if neuron_clique
    "neuron_idx": 10, # if neuron_split
    "num_splits": 2 # if neuron_split
}

### **********************Distillation attack************************
layer_name="fc2"
attack_type="logits"
epoch_attack = 20
save_path_attack = "l_" + layer_name + "_attack_type_" + str(attack_type) + "_epochs_" + str(epoch_attack)

cf_mlp_attack_distillation = {
    "configuration": cf_mlp_dict,
    "database": cf_mlp_dict["database"],
    "path_model": cf_mlp_attack_dummy_neurons["save_path"],
    "save_path": "results/attacks/distillation/res_encrypt/" + cf_mlp_dict[
        "architecture"].lower() + "/" + save_path_attack + ".pth",
    "device":device,
    "attack_type":attack_type,
    "layer_name": layer_name,
    "epoch_attack": epoch_attack

}

# ************* watermarking CNN architecture ****************
watermark_size = 256
epochs_embed = 1000
layer_name = "fc1.weight"
epoch_check = 30
save_path_embed = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epochs_embed) + \
                  "_epc" + str(epoch_check)

cf_cnn_embed = {"configuration": cf_cnn_dict,
                "database": cf_cnn_dict["database"],
                "watermark_size": watermark_size,
                "epochs": epochs_embed,
                "epoch_check": epoch_check,

                "lr_MN": 1e-3,
                "wd_MN": 1e-4,
                "lambda_1": 1,
                "lambda_2": 1,
                "lambda_3": 1e-5,
                "expansion_factor": 1,

                "layer_name": layer_name,

                "lr": 1e-3,
                "wd": 1e-4,
                "opt": "Adam",
                "scheduler": "MultiStepLR",
                "architecture": cf_cnn_dict['architecture'],

                "save_path": "results/watermarked_models/enc_resistant/" + cf_cnn_dict['architecture'].lower() + "/"
                             + save_path_embed + ".pth",

                "path_model": cf_cnn_dict["save_path"],

                "momentum": 0,
                "milestones": [2000, 3000],
                "gamma": .1,
                "criterion": cf_cnn_dict["criterion"],
                "device": cf_cnn_dict["device"],
                }

epoch_attack = 50
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack)

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
                    "save_path": "results/attacks/finetuning/res_encrypt/" + cf_cnn_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",
                    "momentum": 0.9,
                    "milestones": [150, 200],
                    "gamma": 0.1,
                    "criterion": cf_cnn_dict["criterion"],
                    "device": cf_cnn_dict["device"],
                    "save_fig_path": "results/attacks/finetuning/res_encrypt/" + cf_cnn_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".png",
                    "x_label": "Partition_res_encrypt_CNN",
                    "y_label": "BER/ACC",
                    "show_acc_epoch": False
                    }

cf_cnn_attack_pr = {"configuration": cf_cnn_dict,
                    "database": cf_cnn_dict["database"],
                    "path_model": cf_cnn_embed["save_path"],
                    "watermark_size": watermark_size,
                    "amount": 0.9,
                    "architecture": cf_cnn_dict['architecture'],
                    "save_path": "results/attacks/pruning/res_encrypt/" + cf_cnn_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",
                    "criterion": cf_cnn_dict["criterion"],
                    "device": cf_cnn_dict["device"],

                    }

layer_name = "fc1.weight"
epoch_attack = 100
watermark_size = 256

epoch_check = 20
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + "_epc" \
                   + str(epoch_check)
cf_cnn_attack_ow = {"configuration": cf_cnn_dict,
                    "database": cf_cnn_dict["database"],
                    "watermark_size": watermark_size,
                    "epochs": epochs_embed,
                    "epoch_check": epoch_check,

                    "lr_MN": 1e-3,
                    "wd_MN": 1e-4,
                    "lambda_1": 1,
                    "lambda_2": 1,
                    "lambda_3": 1e-5,
                    "expansion_factor": 1,

                    "layer_name": layer_name,
                    "lr": 1e-4,
                    "wd": 0,
                    "opt": "Adam",
                    "scheduler": "MultiStepLR",
                    "architecture": cf_cnn_dict["architecture"],

                    "path_model": cf_cnn_embed["save_path"],
                    "save_path": "results/attacks/overwriting/res_encrypt/" + cf_cnn_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",

                    "momentum": 0,
                    "milestones": [1000, 2000],
                    "gamma": 0,
                    "criterion": cf_cnn_dict["criterion"],
                    "device": cf_cnn_dict["device"],
                    "save_fig_path": "results/attacks/overwriting/res_encrypt/" + cf_cnn_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".png",
                    "x_label": "Partition_res_encrypt_cnn" + cf_cnn_dict["architecture"].lower(),
                    "y_label": "BER/ACC",
                    "show_acc_epoch": False
                    }

cf_cnn_attack_pia = {}
# ************* watermarking Resnet18 architecture ****************
watermark_size = 256
epochs_embed = 1000
layer_name = "linear.weight"
epoch_check = 30
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + "_epc" \
                   + str(epoch_check)

cf_resnet18_embed = {"configuration": cf_resnet18_dict,
                     "database": cf_resnet18_dict["database"],
                     "watermark_size": watermark_size,
                     "epochs": epochs_embed,
                     "epoch_check": epoch_check,

                     "lr_MN": 1e-3,
                     "wd_MN": 1e-4,
                     "lambda_1": 1,
                     "lambda_2": 1,
                     "lambda_3": 1e-5,
                     "expansion_factor": 1,

                     "layer_name": layer_name,
                     "lr": 1e-4,
                     "wd": 5e-5,
                     "opt": "Adam",
                     "scheduler": "MultiStepLR",
                     "architecture": cf_resnet18_dict['architecture'],

                     "save_path": "results/watermarked_models/res_encrypt/" + cf_resnet18_dict[
                         'architecture'].lower() + "/" + save_path_embed + ".pth",

                     # relu but change in mlp cd
                     "path_model": cf_resnet18_dict["save_path"],
                     "momentum": 0,
                     "milestones": [100, 150],
                     "gamma": 0,
                     "criterion": cf_resnet18_dict["criterion"],
                     "device": cf_resnet18_dict["device"],
                     }

epoch_attack = 50
save_path_attack = layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack)

cf_resnet18_attack_ft = {"configuration": cf_resnet18_dict,
                         "database": cf_resnet18_dict["database"],
                         "path_model": cf_resnet18_embed['save_path'],
                         "epochs": epoch_attack,
                         "watermark_size": watermark_size,

                         "layer_name": layer_name,
                         "lr": 1e-4,
                         "wd": 0,
                         "scheduler": "MultiStepLR",
                         "opt": "Adam",
                         "architecture": cf_resnet18_dict["architecture"],
                         "save_path": "results/attacks/finetuning/res_encrypt/" + cf_resnet18_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",
                         "momentum": 0,
                         "milestones": [150, 200],
                         "gamma": 0,
                         "criterion": cf_resnet18_dict["criterion"],
                         "device": cf_resnet18_dict["device"],
                         "save_fig_path": "results/attacks/finetuning/res_encrypt/" + cf_resnet18_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".png",
                         "x_label": "Partition_res_encrypt_resnet18",
                         "y_label": "BER/ACC",
                         "show_acc_epoch": False
                         }

cf_resnet18_attack_pr = {"configuration": cf_resnet18_dict,
                         "database": cf_resnet18_dict["database"],
                         "path_model": cf_resnet18_embed["save_path"],
                         "watermark_size": watermark_size,
                         "criterion": cf_resnet18_dict["criterion"],
                         "device": cf_resnet18_dict["device"],
                         "architecture": cf_resnet18_dict["architecture"],
                         "save_path": "results/attacks/pruning/res_encrypt/" + cf_resnet18_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",

                         }
watermark_size = 256
layer_name = "linear.weight"
epoch_attack = 100
epoch_check = 20
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + "_epc" \
                   + str(epoch_check)

cf_resnet18_attack_ow = {"configuration": cf_resnet18_dict,
                         "database": cf_resnet18_dict["database"],
                         "path_model": cf_resnet18_embed["save_path"],
                         "watermark_size": watermark_size,
                         "epochs": epoch_attack,
                         "epoch_check": epoch_check,

                         "lr_MN": 1e-3,
                         "wd_MN": 1e-4,
                         "lambda_1": 1,
                         "lambda_2": 1,
                         "lambda_3": 1e-5,
                         "expansion_factor": 1,

                         "layer_name": layer_name,
                         "lr": 1e-4,
                         "wd": 0,
                         "opt": "Adam",
                         "scheduler": "MultiStepLR",
                         "architecture": cf_resnet18_dict["architecture"],
                         "save_path": "results/attacks/overwriting/res_encrypt/" + cf_resnet18_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",

                         "momentum": 0,
                         "milestones": [1000, 2000],
                         "gamma": 0,
                         "criterion": cf_resnet18_dict["criterion"],
                         "device": cf_resnet18_dict["device"],
                         "save_fig_path": "results/attacks/overwriting/res_encrypt/" + cf_resnet18_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".png",
                         "x_label": "Partition_res_encrypt" + cf_resnet18_dict["architecture"].lower(),
                         "y_label": "BER/ACC",
                         "show_acc_epoch": False
                         }

cf_resnet18_attack_pia = {}

# ************* watermarking MLP_RIGA architecture ****************
watermark_size = 256
epochs_embed = 10000
layer_name = "fc2.weight"  # "conv2.weight"
epoch_check = 20
save_path_embed = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epochs_embed) + "_epc" + str(
    epoch_check)

cf_mlp_riga_embed = {"configuration": cf_mlp_riga_dict,
                     "database": cf_mlp_riga_dict["database"],
                     "watermark_size": watermark_size,
                     "epochs": epochs_embed,
                     "epoch_check": epoch_check,

                     "lambda_1": 1,
                     "lambda_2": 1,
                     "lambda_3": 1e-2,
                     "expansion_factor": 2,

                     "lr_MN": 1e-2,
                     "wd_MN": 1e-3,

                     "layer_name": layer_name,
                     "lr": 1e-3,
                     "wd": 1e-4,
                     "opt": "Adam",
                     "scheduler": "MultiStepLR",
                     "architecture": cf_mlp_riga_dict["architecture"],

                     "path_model": cf_mlp_riga_dict["save_path"],
                     "save_path": "results/watermarked_models/res_encrypt/" + cf_mlp_riga_dict[
                         'architecture'].lower() + "/"
                                  + save_path_embed + ".pth",

                     "momentum": 0,
                     "milestones": [30, 150],
                     "gamma": .1,
                     "criterion": cf_mlp_riga_dict["criterion"],
                     "device": cf_mlp_riga_dict["device"],
                     }

epoch_attack = 50
lr_attack = 1e-4
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack)

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
                         "save_path": "results/attacks/finetuning/res_encrypt/" + cf_mlp_riga_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",

                         "save_fig_path": "results/attacks/finetuning/res_encrypt/" + cf_mlp_riga_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".png",
                         "x_label": "Partition_enc_resistant_" + cf_mlp_riga_dict["architecture"].lower(),
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
                         "save_path": "results/attacks/pruning/res_encrypt/" + cf_mlp_riga_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",
                         }

layer_name = "fc2.weight"
epoch_attack = 100
watermark_size = 256
epoch_check = 20
save_path_attack = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + "_epc" \
                   + str(epoch_check)

cf_mlp_riga_attack_ow = {"configuration": cf_mlp_riga_dict,
                         "database": cf_mlp_riga_dict["database"],
                         "watermark_size": watermark_size,
                         "epochs": epoch_attack,
                         "epoch_check": epoch_check,

                         "lr_MN": 1e-3,
                         "wd_MN": 1e-4,
                         "lambda_1": 1,
                         "lambda_2": 1,
                         "lambda_3": 1e-5,
                         "expansion_factor": 2,

                         "layer_name": layer_name,
                         "lr": 1e-3,
                         "wd": 1e-4,
                         "opt": "Adam",
                         "scheduler": "MultiStepLR",
                         "architecture": cf_mlp_riga_dict["architecture"],

                         "save_path": "results/attacks/overwriting/res_encrypt/" + cf_mlp_riga_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",

                         "momentum": 0,
                         "milestones": [30, 2000],
                         "gamma": .1,
                         "criterion": cf_mlp_riga_dict["criterion"],
                         "device": cf_mlp_riga_dict["device"],
                         "save_fig_path": "results/attacks/overwriting/res_encrypt/" + cf_mlp_riga_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".png",
                         "x_label": "Partition_res_encrypt" + cf_mlp_riga_dict["architecture"].lower(),
                         "y_label": "BER/ACC",
                         "show_acc_epoch": False
                         }

cf_mlp_riga_attack_pia = {}
