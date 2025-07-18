from configs.cf_train.cf_mlp import cf_mlp_dict
from configs.cf_train.cf_cnn import cf_cnn_dict
from configs.cf_train.cf_mlp_riga import cf_mlp_riga_dict
from configs.cf_train.cf_resnet18 import cf_resnet18_dict

from copy import deepcopy

#********************************************************************
#************** watermarking MLP architecture ***********************
#********************************************************************
method_wat = "diction"
watermark_size = 256
epochs_embed = 10000
epoch_check = 30
layer_name = ["fc2"]
save_path_embed = "_l" + layer_name[0] + "_wat" + str(watermark_size) + "_ep" + str(epochs_embed) + "_epc" + \
                  str(epoch_check)
cf_mlp_embed = {"configuration": cf_mlp_dict,  # the configuration of the target model
                "database": cf_mlp_dict["database"],  # the database to train the target model
                "watermark_size": watermark_size,  # the size of the watermark to be embedded
                "epochs": epochs_embed,  # the maximum number of epochs to train the target model
                "epoch_check": epoch_check,  # the number of epochs to check the watermarking process

                # Latent space parameters
                "batch_size": int(cf_mlp_dict["batch_size"] * 1.),  # the number of samples of trigger set in a batch

                "mean": 0.,  # the mean of the latent space
                "std": 1.,  # the standard deviation of the latent space

                "square_size": 5,  # the size of the square to be added to the trigger set
                "start_x": 0,  # the x coordinate of the square to be added to the trigger set
                "start_y": 0,  # the y coordinate of the square to be added to the trigger set
                "square_value": 1,  # the value of the square to be added to the trigger set

                "n_features": 1.,  # the percentage of features to be used in the watermarking process

                "wd_proj": 1e-4,  # the weight decay to train the projection model
                "lr_proj": 1e-3,  # the learning rate to train the projection model


                # the regularization parameter of the  distance between the original model and
                # the watermarked model
                "lambda_mse": 0,
                "lambda_proj": 1,
                "lambda_acts": 1,

                "layer_name": layer_name,  # the layer name to be watermarked
                "wd": 1e-4,  # weight decay to train the target model
                "lr": 1e-3,  # learning rate to train the target model
                "opt": "Adam",  # optimizer
                "scheduler": "MultiStepLR",  # the scheduler to update the learning rate
                "architecture": cf_mlp_dict["architecture"],  # the architecture of the target model
                "momentum": 0,  # momentum to train the target model (if needed)
                "milestones": [20, 60],  # the milestones to update the learning rate
                "gamma": 0.1,  # the factor to update the learning rate
                "criterion": cf_mlp_dict["criterion"],  # the criterion to train the target model
                "device": cf_mlp_dict["device"],  # the device to train the target model

                "path_model": cf_mlp_dict["save_path"],  # Path of the original model
                "save_path": "results/watermarked_models/" + method_wat + "/" + cf_mlp_dict['architecture'].lower() + "/"
                             + save_path_embed + ".pth"  # Path to save the target/watermarked model

                }

#### *********************************Fine tuning attack************************************
epoch_attack = 50
save_path_attack = "_l" + layer_name[0] + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + ".pth"
cf_mlp_attack_ft = {"configuration": cf_mlp_dict,
                    "database": cf_mlp_dict["database"],
                    "watermark_size": watermark_size,
                    "epochs": epoch_attack,
                    "lr": 1e-4,
                    "wd": 0,
                    "opt": "Adam",
                    "scheduler": "MultiStepLR",
                    "architecture": cf_mlp_dict["architecture"],

                    "path_model": cf_mlp_embed["save_path"],
                    "save_path": "results/attacks/finetuning/" + method_wat + "/" + cf_mlp_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",
                    "momentum": 0,
                    "milestones": [100, 2000],
                    "gamma": 0,
                    "criterion": cf_mlp_dict["criterion"],
                    "device": cf_mlp_dict["device"],
                    "save_fig_path": "results/attacks/finetuning/" + method_wat + "/" + cf_mlp_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".png",
                    "x_label": "Partition_diction_" + cf_mlp_dict["architecture"].lower(),
                    "y_label": "BER/ACC",
                    "show_acc_epoch": False
                    }

### *********************************Pruning attack****************************************
cf_mlp_attack_pr = {"configuration": cf_mlp_dict,
                    "database": cf_mlp_dict["database"],
                    "path_model": cf_mlp_embed["save_path"],
                    "watermark_size": watermark_size,
                    "criterion": cf_mlp_dict["criterion"],
                    "save_path": "results/attacks/pruning/" + method_wat + "/" + cf_mlp_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",
                    "architecture": cf_mlp_dict["architecture"],
                    "device": cf_mlp_dict["device"],
                    }

### *********************************Overwriting attack************************************
epoch_attack = 1000
watermark_size = 256
epoch_check = 30
layer_name = ["fc2"]
save_path_attack = "_l" + layer_name[0] + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) \
                   + "_epc" + str(epoch_check)
cf_mlp_attack_ow = {"configuration": cf_cnn_dict,
                    "database": cf_mlp_dict["database"],
                    "path_model": cf_mlp_embed["save_path"],
                    "watermark_size": watermark_size,

                    # Latent space parameters
                    "mean": 0,
                    "std": 1,
                    "n_features": 1,

                    "square_size": 5,
                    "start_x": 10,
                    "start_y": 10,
                    "square_value": 1,

                    "batch_size": int(cf_mlp_dict["batch_size"] * 1.),
                    "epochs": epoch_attack,
                    "epoch_check": epoch_check,

                    "wd_proj": 1e-4,  # the weight decay to train the projection model
                    "lr_proj": 1e-3,  # the learning rate to train the projection model

                    "lambda_mse": 0,
                    # the regularization parameter of the  distance between the original model and
                    # the watermarked model
                    "lambda_proj": 1,
                    "lambda_acts": 1,

                    "layer_name": layer_name,

                    "lr": 1e-3,
                    "wd": 1e-4,
                    "scheduler": "MultiStepLR",
                    "architecture": cf_mlp_dict["criterion"],
                    "save_path": "results/attacks/overwriting/" + method_wat + "/" + cf_mlp_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",
                    "momentum": 0,
                    "milestones": [20, 10],
                    "gamma": 0.1,
                    "criterion": cf_mlp_dict["criterion"],
                    "device": cf_mlp_dict["device"],
                    "save_fig_path": "results/attacks/overwriting/" + method_wat + "/" + cf_mlp_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".png",
                    "x_label": "Partition_diction_" + cf_mlp_dict["architecture"].lower(),
                    "y_label": "BER/ACC",
                    "show_acc_epoch": False
                    }

### *********************************PIA attack**********************************************
cf_non_watermarked = deepcopy(cf_mlp_dict)
cf_non_watermarked["epochs"] = 2
cf_non_watermarked["show_acc_epoch"] = False
cf_watermarked = deepcopy(cf_mlp_embed)
cf_watermarked["epochs"] = 10
cf_watermarked["show_acc_epoch"] = False
nb_examples = 800
save_path_attack = "_l" + layer_name[0] + "_wat" + str(
    cf_watermarked["watermark_size"]) + "_ep" + str(cf_non_watermarked["epochs"]) + "_nb_examples" + str(nb_examples)
cf_mlp_attack_pia = {"train_non_watermarked": cf_non_watermarked,
                     "train_watermarked": cf_watermarked,
                     "save_path": "results/attacks/pia/" + method_wat + "/" + cf_mlp_dict[
                         "architecture"].lower() + "/" + save_path_attack + ".pth",
                     "nb_examples": nb_examples,

                     }

### *********************************Dummy neurons attack***************************************
layer_name="fc2"
num_dummy=2
attack_type="neuron_clique"
save_path_attack = "l_" + layer_name + "_attack_type_" + str(attack_type)

cf_mlp_attack_dummy_neurons = {
    "configuration": cf_mlp_dict,
    "database": cf_mlp_dict["database"],
    "path_model": cf_mlp_embed["save_path"],
    "save_path": "results/attacks/dummy_neurons/" + method_wat + "/" + cf_mlp_dict[
        "architecture"].lower() + "/" + save_path_attack + ".pth",
    "device":cf_mlp_dict["device"],
    "attack_type":attack_type,
    "layer_name": layer_name,
    "num_dummy": 2, # if neuron_clique
    "neuron_idx": 10, # if neuron_split
    "num_splits": 2 # if neuron_split
}

### *********************************Distillation attack************************
layer_name="fc2"
attack_type="logits"
epoch_attack = 20
save_path_attack = "l_" + layer_name + "_attack_type_" + str(attack_type) + "_epochs_" + str(epoch_attack)

cf_mlp_attack_distillation = {
    "configuration": cf_mlp_dict,
    "database": cf_mlp_dict["database"],
    "path_model": cf_mlp_attack_dummy_neurons["save_path"],
    "save_path": "results/attacks/distillation/" + method_wat + "/" + cf_mlp_dict[
        "architecture"].lower() + "/" + save_path_attack + ".pth",
    "device":cf_mlp_dict["device"],
    "attack_type":attack_type,
    "layer_name": layer_name,
    "epoch_attack": epoch_attack

}

#********************************************************************
#************** watermarking CNN architecture ***********************
#********************************************************************

watermark_size = 256
epochs_embed = 10000
layer_name = ["fc1"]  # fc1
epoch_check = 20
save_path_embed = "_l" + layer_name[0] + "_wat" + str(watermark_size) + "_ep" + str(epochs_embed) + "_epc" + \
                  str(epoch_check)

cf_cnn_embed = {"configuration": cf_cnn_dict,
                "database": cf_cnn_dict["database"],
                "watermark_size": watermark_size,
                "epochs": epochs_embed,
                "epoch_check": epoch_check,

                # Latent space parameters
                "batch_size": int(cf_cnn_dict["batch_size"] * 1.),
                "mean": 0,
                "std": 1,

                "square_size": 5,
                "start_x": 0,
                "start_y": 0,
                "square_value": 1,

                "layer_name": layer_name,
                "n_features": 1.,

                "lr_proj": 1e-3,  # the learning rate to train the projection model
                "wd_proj": 1e-4,  # the weight decay to train the projection model

                "lambda_mse": 0,  # the regularization parameter of the  distance between the original model and
                # the watermarked model
                "lambda_proj": 1.,
                "lambda_acts": 1.,

                "lr": 1e-3,
                "wd": 1e-4,
                "opt": "Adam",
                "scheduler": "MultiStepLR",
                "architecture": cf_cnn_dict['architecture'],
                "momentum": 0,
                "milestones": [10, 3000],
                "gamma": .1,
                "criterion": cf_cnn_dict["criterion"],
                "device": cf_cnn_dict["device"],

                "path_model": cf_cnn_dict["save_path"],
                "save_path": "results/watermarked_models/" + method_wat + "/" + cf_cnn_dict['architecture'].lower() + "/"
                             + save_path_embed + ".pth",
                }

#### *********************************Fine tuning attack*************************************
epoch_attack = 50
save_path_attack = "_l" + layer_name[0] + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + "_epc" + \
                   str(epoch_check)
cf_cnn_attack_ft = {"configuration": cf_cnn_dict,
                    "database": cf_cnn_dict["database"],
                    "path_model": cf_cnn_embed['save_path'],
                    "epochs": epoch_attack,
                    "watermark_size": watermark_size,

                    "lr": 1e-4,
                    "wd": 1e-5,
                    "scheduler": "MultiStepLR",
                    "opt": "Adam",
                    "architecture": cf_cnn_dict["architecture"],

                    "momentum": 0,
                    "milestones": [50, 200],
                    "gamma": 0.1,
                    "criterion": cf_cnn_dict["criterion"],
                    "device": cf_cnn_dict["device"],

                    "save_path": "results/attacks/finetuning/" + method_wat + "/" + cf_cnn_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",
                    "save_fig_path": "results/attacks/finetuning/" + method_wat + "/" + cf_cnn_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".png",

                    "x_label": "Partition_diction_CNN",
                    "y_label": "BER/ACC",
                    "show_acc_epoch": False
                    }

#### *********************************Pruning attack*****************************************
cf_cnn_attack_pr = {"configuration": cf_cnn_dict,
                    "database": cf_cnn_dict["database"],
                    "path_model": cf_cnn_embed["save_path"],
                    "watermark_size": watermark_size,
                    "criterion": cf_cnn_dict["criterion"],
                    "architecture": cf_cnn_dict["architecture"],
                    "save_path": "results/attacks/pruning/" + method_wat + "/" + cf_cnn_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",
                    "device": cf_cnn_dict["device"],
                    }

#### *********************************Overwriting attack*************************************
layer_name = ["fc1"]
epoch_attack = 30
epoch_check = 20
watermark_size = 256
save_path_attack = "_l" + layer_name[0] + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) \
                   + "_epc" + str(epoch_check)
cf_cnn_attack_ow = {"configuration": cf_cnn_dict,
                    "database": cf_cnn_dict["database"],
                    "path_model": cf_cnn_embed["save_path"],
                    "watermark_size": watermark_size,

                    # Latent space parameters
                    "mean": 0,
                    "std": 1,

                    "square_size": 5,
                    "start_x": 10,
                    "start_y": 10,
                    "square_value": 1,

                    "n_features": 1.,

                    "batch_size": int(cf_cnn_dict["batch_size"] * 0.5),
                    "epochs": epoch_attack,
                    "epoch_check": epoch_check,

                    # Training parameters
                    "wd_proj": 1e-4,  # the weight decay to train the projection model
                    "lr_proj": 1e-3,  # the learning rate to train the projection model

                    "lambda_mse": 0,  # the regularization parameter of the  distance between the original model and
                    # the watermarked model
                    "lambda_proj": 1,
                    "lambda_acts": 1,

                    "layer_name": layer_name,

                    "lr": 1e-3,
                    "wd": 1e-4,

                    "opt": "Adam",
                    "scheduler": "MultiStepLR",
                    "architecture": cf_cnn_dict["criterion"],

                    "momentum": 0,
                    "milestones": [10, 2000],
                    "gamma": 0.1,
                    "criterion": cf_cnn_dict["criterion"],
                    "device": cf_cnn_dict["device"],

                    "save_path": "results/attacks/overwriting/" + method_wat + "/" + cf_cnn_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".pth",
                    "save_fig_path": "results/attacks/overwriting/" + method_wat + "/" + cf_cnn_dict[
                        "architecture"].lower() + "/" + save_path_attack + ".png",
                    "x_label": "Partition_diction_" + cf_cnn_dict["architecture"].lower(),
                    "y_label": "BER/ACC",
                    "show_acc_epoch": False
                    }

#### *********************************PIA attack*************************************
cf_cnn_attack_pia = {"train_non_watermarked": cf_non_watermarked,
                     "train_watermarked": cf_watermarked,
                     "save_path": "results/attacks/pia/" + method_wat + "/" + cf_cnn_dict[
                         "architecture"].lower() + "/" + save_path_attack + ".pth",
                     "nb_examples": nb_examples
                     }

### ****************************Dummy neurons attack***************************************
layer_name="fc1"
num_dummy=2
attack_type="neuron_clique"
save_path_attack = "l_" + layer_name + "_attack_type_" + str(attack_type)

cf_cnn_attack_dummy_neurons = {
    "configuration": cf_cnn_dict,
    "database": cf_cnn_dict["database"],
    "path_model": cf_cnn_embed["save_path"],
    "save_path": "results/attacks/dummy_neurons/" + method_wat + "/" + cf_cnn_dict[
        "architecture"].lower() + "/" + save_path_attack + ".pth",
    "device":cf_cnn_dict["device"],
    "attack_type":attack_type,
    "layer_name": layer_name,
    "num_dummy": 2, # if neuron_clique
    "neuron_idx": 10, # if neuron_split
    "num_splits": 2 # if neuron_split
}

### **********************Distillation attack************************
layer_name="fc1"
attack_type="logits"
epoch_attack = 20
save_path_attack = "l_" + layer_name + "_attack_type_" + str(attack_type) + "_epochs_" + str(epoch_attack)

cf_cnn_attack_distillation = {
    "configuration": cf_cnn_dict,
    "database": cf_cnn_dict["database"],
    "path_model": cf_cnn_attack_dummy_neurons["save_path"],
    "save_path": "results/attacks/distillation/" + method_wat + "/" + cf_mlp_dict[
        "architecture"].lower() + "/" + save_path_attack + ".pth",
    "device":cf_cnn_dict["device"],
    "attack_type":attack_type,
    "layer_name": layer_name,
    "epoch_attack": epoch_attack

}

#********************************************************************
#************** watermarking Resnet18 architecture ******************
#********************************************************************
watermark_size = 256
epochs_embed = 1000
epoch_check = 30
layer_name = ["view"]
# layer1.0.conv1', layer3.1.conv1, view
save_path_embed = "_l" + layer_name[0] + "_wat" + str(watermark_size) + "_ep" + str(epochs_embed) + "_epc" + \
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

                     "batch_size": int(cf_resnet18_dict["batch_size"] * 1.),
                     "n_features": 1.,

                     "lr_proj": 1e-3,  # the learning rate to train the projection model
                     "wd_proj": 1e-4,  # the weight decay to train the projection model

                     "lambda_mse": 0,  # the regularization parameter of the  distance between the original model and
                     # the watermarked model
                     "lambda_proj": 1,
                     "lambda_acts": 1,

                     "layer_name": layer_name,

                     "lr": 1e-3,
                     "wd": 1e-4,

                     "opt": "Adam",
                     "scheduler": "MultiStepLR",
                     "architecture": cf_resnet18_dict['architecture'],
                     "save_path": "results/watermarked_models/" + method_wat + "/" + cf_resnet18_dict[
                         'architecture'].lower() + "/" + save_path_embed + ".pth",
                     # relu but change in mlp cd
                     "path_model": cf_resnet18_dict["save_path"],
                     "momentum": 0,
                     "milestones": [20, 150],
                     "gamma": .1,
                     "criterion": cf_resnet18_dict["criterion"],
                     "device": cf_resnet18_dict["device"]
                     }

#### *********************************Fine tuning attack*************************************
epoch_attack = 50
save_path_attack = "_l" + layer_name[0] + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + "_epc" + \
                   str(epoch_check)
cf_resnet18_attack_ft = {"configuration": cf_resnet18_dict,
                         "database": cf_resnet18_dict["database"],
                         "path_model": cf_resnet18_embed['save_path'] + '.pth',
                         "epochs": epoch_attack,
                         "watermark_size": watermark_size,

                         "layer_name": layer_name,
                         "lr": 1e-4,
                         "wd": 1e-5,
                         "scheduler": "MultiStepLR",
                         "opt": "Adam",
                         "architecture": cf_resnet18_dict["architecture"],
                         "save_path": "results/attacks/finetuning/" + method_wat + "/" + cf_resnet18_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",
                         "momentum": 0,
                         "milestones": [20, 200],
                         "gamma": 0.1,
                         "criterion": cf_resnet18_dict["criterion"],
                         "device": cf_resnet18_dict["device"],
                         "save_fig_path": "results/attacks/finetuning/" + method_wat + "/" + cf_resnet18_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".png",
                         "x_label": "Partition_diction_resnet18",
                         "y_label": "BER/ACC",
                         "show_acc_epoch": False
                         }

#### *********************************Pruning attack*****************************************
cf_resnet18_attack_pr = {"configuration": cf_resnet18_dict,
                         "database": cf_resnet18_dict["database"],
                         "path_model": cf_resnet18_embed["save_path"] + ".pth",
                         "watermark_size": watermark_size,
                         "criterion": cf_resnet18_dict["criterion"],
                         "architecture": cf_resnet18_dict["architecture"],
                         "save_path": "results/attacks/pruning/" + method_wat + "/" + cf_resnet18_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",
                         "device": cf_resnet18_dict["device"],
                         }

#### *********************************Overwriting attack*************************************
layer_name = ["view"]
# layer1.0.conv1', layer3.1.conv1, view
epoch_attack = 100
watermark_size = 256
epoch_check = 20
save_path_attack = "_l" + layer_name[0] + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) \
                   + "_epc" + str(epoch_check)
cf_resnet18_attack_ow = {"configuration": cf_resnet18_dict,
                         "database": cf_resnet18_dict["database"],
                         "path_model": cf_resnet18_embed["save_path"] + ".pth",
                         "watermark_size": watermark_size,
                         # Latent space parameters
                         "mean": 0,
                         "std": 1,
                         "square_size": 5,
                         "start_x": 10,
                         "start_y": 10,
                         "square_value": 1,

                         "n_features": 1.,
                         "batch_size": int(cf_resnet18_dict["batch_size"] * 1.),
                         "epochs": epoch_attack,
                         "epoch_check": epoch_check,

                         # Training parameters
                         "lr_proj": 1e-3,  # the learning rate to train the projection model
                         "wd_proj": 1e-4,  # the weight decay to train the projection model

                         "lambda_mse": 0,  # the regularization parameter of the  distance between the original model and
                         # the watermarked model
                         "lambda_proj": 1,
                         "lambda_acts": 1,

                         "layer_name": layer_name,
                         "lr": 1e-3,
                         "wd": 1e-4,
                         "opt": "Adam",
                         "scheduler": "MultiStepLR",
                         "architecture": cf_resnet18_dict["criterion"],
                         "save_path": "results/attacks/overwriting/" + method_wat + "/" + cf_resnet18_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",
                         "momentum": 0,
                         "milestones": [1000, 2000],
                         "gamma": 0,
                         "criterion": cf_resnet18_dict["criterion"],
                         "device": cf_resnet18_dict["device"],
                         "save_fig_path": "results/attacks/overwriting/" + method_wat + "/" + cf_resnet18_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".png",
                         "x_label": "Partition_diction_" + cf_resnet18_dict["architecture"].lower(),
                         "y_label": "BER/ACC",
                         "show_acc_epoch": False
                         }

#### *********************************PIA attack*********************************************
cf_resnet18_attack_pia = {"train_non_watermarked": cf_non_watermarked,
                          "train_watermarked": cf_watermarked,
                          "save_path": "results/attacks/pia/" + method_wat + "/" + cf_resnet18_dict[
                              "architecture"].lower() + "/" + save_path_attack + ".pth",
                          "nb_examples": nb_examples
                          }

### ****************************Dummy neurons attack*****************************************
layer_name="linear"
num_dummy=2
attack_type="neuron_clique"
save_path_attack = "l_" + layer_name + "_attack_type_" + str(attack_type)
cf_resnet18_attack_dummy_neurons = {
    "configuration": cf_resnet18_dict,
    "database": cf_resnet18_dict["database"],
    "path_model": cf_resnet18_embed["save_path"],
    "save_path": "results/attacks/dummy_neurons/" + method_wat + "/" + cf_resnet18_dict[
        "architecture"].lower() + "/" + save_path_attack + ".pth",
    "device":cf_resnet18_dict["device"],
    "attack_type":attack_type,
    "layer_name": layer_name,
    "num_dummy": 2, # if neuron_clique
    "neuron_idx": 10, # if neuron_split
    "num_splits": 2 # if neuron_split
}

### ****************************Distillation attack****************************************
layer_name="linear"
attack_type="logits"
epoch_attack = 20
save_path_attack = "l_" + layer_name + "_attack_type_" + str(attack_type) + "_epochs_" + str(epoch_attack)

cf_resnet18_attack_distillation = {
    "configuration": cf_resnet18_dict,
    "database": cf_resnet18_dict["database"],
    "path_model": cf_resnet18_attack_dummy_neurons["save_path"],
    "save_path": "results/attacks/distillation/" + method_wat + "/" + cf_resnet18_dict[
        "architecture"].lower() + "/" + save_path_attack + ".pth",
    "device":cf_resnet18_dict["device"],
    "attack_type":attack_type,
    "layer_name": layer_name,
    "epoch_attack": epoch_attack

}

#********************************************************************
#************** watermarking MLP_RIGA architecture *******************
#********************************************************************

watermark_size = 256
epochs_embed = 101
layer_name = ["fc2"]  # "conv1", "flatten", "fc2"
epoch_check = 30
save_path_embed = "_l" + layer_name[0] + "_wat" + str(watermark_size) + "_ep" + str(epochs_embed) + "_epc" + \
                  str(epoch_check)

cf_mlp_riga_embed = {"configuration": cf_mlp_riga_dict,
                     "database": cf_mlp_riga_dict["database"],
                     "watermark_size": watermark_size,
                     "epochs": epochs_embed,
                     "epoch_check": epoch_check,

                     # Latent space parameters
                     "batch_size": int(cf_mlp_riga_dict["batch_size"] * 1.),
                     "mean": 0,
                     "std": 1,
                     # the size of the square to be added to the trigger set and its position
                     "square_size": 5,
                     "start_x": 0,
                     "start_y": 0,
                     "square_value": 1,

                     "n_features": 1.,  # the percentage of features to be used in the watermarking process

                     "wd_proj": 1e-3,  # the weight decay to train the projection model
                     "lr_proj": 1e-2,  # the learning rate to train the projection model

                     "lambda_mse": 0,
                     # the regularization parameter of the  distance between the original model and
                     # the watermarked model
                     "lambda_proj": 1,
                     "lambda_acts": 1,

                     "layer_name": layer_name,
                     "lr": 1e-3,
                     "wd": 1e-4,

                     "opt": "Adam",
                     "scheduler": "MultiStepLR",
                     "architecture": cf_mlp_riga_dict["architecture"],

                     "path_model": cf_mlp_riga_dict["save_path"],
                     "save_path": "results/watermarked_models/" + method_wat + "/" + cf_mlp_riga_dict['architecture'].lower() + "/"
                                  + save_path_embed + ".pth",
                     "momentum": 0,
                     "milestones": [20, 1000],
                     "gamma": 0.1,
                     "criterion": cf_mlp_riga_dict["criterion"],
                     "device": cf_mlp_riga_dict["device"]
                     }

#### *********************************Fine tuning attack************************************
epoch_attack = 50
save_path_attack = "_l" + layer_name[0] + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) + "_epc" + \
                   str(epoch_check)
cf_mlp_riga_attack_ft = {"configuration": cf_mlp_riga_embed,
                         "database": cf_mlp_riga_embed["database"],
                         "watermark_size": watermark_size,
                         "epochs": epoch_attack,
                         "lr": 1e-4,
                         "wd": 0,
                         "opt": "Adam",
                         "scheduler": "MultiStepLR",
                         "architecture": cf_mlp_riga_dict["architecture"],

                         "path_model": cf_mlp_riga_embed["save_path"],
                         "save_path": "results/attacks/finetuning/" + method_wat + "/" + cf_mlp_riga_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",
                         "momentum": 0,
                         "milestones": [20, 50],
                         "gamma": 0.1,
                         "criterion": cf_mlp_riga_dict["criterion"],
                         "device": cf_mlp_riga_dict["device"],
                         "save_fig_path": "results/attacks/finetuning/" + method_wat + "/" + cf_mlp_riga_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".png",
                         "x_label": "Partition_diction_" + cf_mlp_riga_dict["architecture"].lower(),
                         "y_label": "BER/ACC",
                         "show_acc_epoch": False
                         }

#### *********************************Pruning attack************************************
cf_mlp_riga_attack_pr = {"configuration": cf_mlp_riga_dict,
                         "database": cf_mlp_riga_dict["database"],
                         "path_model": cf_mlp_riga_embed["save_path"],
                         "watermark_size": watermark_size,
                         "criterion": cf_mlp_riga_dict["criterion"],
                         "save_path": "results/attacks/pruning/" + method_wat + "/" + cf_mlp_riga_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",
                         "architecture": cf_mlp_riga_dict["architecture"],
                         "device": cf_mlp_riga_dict["device"],
                         }

#### *********************************overwriting attack************************************
epoch_attack = 40
watermark_size = 512
epoch_check = 10
save_path_attack = "_l" + layer_name[0] + "_wat" + str(watermark_size) + "_ep" + str(epoch_attack) \
                   + "_epc" + str(epoch_check)
cf_mlp_riga_attack_ow = {"configuration": cf_mlp_riga_embed,
                         "database": cf_mlp_riga_embed["database"],
                         "path_model": cf_mlp_riga_embed["save_path"],
                         "watermark_size": watermark_size,

                         # Latent space parameters
                         "mean": .0,
                         "std": 1.,

                         "square_size": 5,
                         "start_x": 10,
                         "start_y": 10,
                         "square_value": 1,

                         "n_features": 1.,
                         "batch_size": int(cf_mlp_riga_dict["batch_size"] * 0.5),

                         "epochs": epoch_attack,
                         "epoch_check": epoch_check,

                         # Training parameters
                         "wd_proj": 1e-4,  # the weight decay to train the projection model
                         "lr_proj": 1e-3,  # the learning rate to train the projection model

                         "lambda_mse": 0,
                         # the regularization parameter of the  distance between the original model and
                         # the watermarked model
                         "lambda_proj": 1,
                         "lambda_acts": 1,


                         "layer_name": layer_name,
                         "lr": 1e-3,
                         "wd": 1e-4,
                         "scheduler": "MultiStepLR",
                         "architecture": cf_mlp_riga_dict["criterion"],
                         "save_path": "results/attacks/overwriting/" + method_wat + "/" + cf_mlp_riga_dict[
                             "architecture"].lower() + "/" + save_path_attack + ".pth",
                         "momentum": 0,
                         "milestones": [20, 30],
                         "gamma": 0.1,
                         "criterion": cf_mlp_riga_embed["criterion"],
                         "device": cf_mlp_riga_embed["device"],
                         "save_fig_path": "results/attacks/overwriting/" + method_wat + "/" + cf_mlp_riga_embed[
                             "architecture"].lower() + "/" + save_path_attack + ".png",
                         "x_label": "Partition_diction_" + cf_mlp_riga_embed["architecture"].lower(),
                         "y_label": "BER/ACC",
                         "show_acc_epoch": False
                         }

#### *********************************PIA attack************************************
cf_non_watermarked = deepcopy(cf_mlp_riga_dict)
cf_non_watermarked["epochs"] = 2
cf_non_watermarked["show_acc_epoch"] = False
cf_watermarked = deepcopy(cf_mlp_riga_embed)
cf_watermarked["epochs"] = 10
cf_watermarked["show_acc_epoch"] = False
nb_examples = 1224
save_path_attack = "_l" + layer_name[0] + "_wat" + str(
    cf_watermarked["watermark_size"]) + "_ep" + str(cf_non_watermarked["epochs"]) + "_nb_examples" + str(nb_examples)

cf_mlp_riga_attack_pia = {"train_non_watermarked": cf_non_watermarked,
                          "train_watermarked": cf_watermarked,
                          "save_path": "results/attacks/pia/" + method_wat + "/" + cf_mlp_riga_dict[
                              "architecture"].lower() + "/" + save_path_attack + ".pth",
                          "nb_examples": nb_examples
                          }

### ****************************Dummy neurons attack***************************************
layer_name="fc1"
num_dummy=2
attack_type="neuron_clique"
save_path_attack = "l_" + layer_name + "_attack_type_" + str(attack_type)

cf_mlp_riga_attack_dummy_neurons = {
    "configuration": cf_mlp_riga_dict,
    "database": cf_mlp_riga_dict["database"],
    "path_model": cf_mlp_riga_embed["save_path"],
    "save_path": "results/attacks/dummy_neurons/" + method_wat + "/" + cf_mlp_riga_dict[
        "architecture"].lower() + "/" + save_path_attack + ".pth",
    "device":cf_mlp_riga_dict["device"],
    "attack_type":attack_type,
    "layer_name": layer_name,
    "num_dummy": 2, # if neuron_clique
    "neuron_idx": 10, # if neuron_split
    "num_splits": 2 # if neuron_split
}

### **********************Distillation attack************************
layer_name="fc1"
attack_type="logits"
epoch_attack = 20
save_path_attack = "l_" + layer_name + "_attack_type_" + str(attack_type) + "_epochs_" + str(epoch_attack)

cf_mlp_riga_attack_distillation = {
    "configuration": cf_mlp_riga_dict,
    "database": cf_mlp_riga_dict["database"],
    "path_model": cf_mlp_riga_attack_dummy_neurons["save_path"],
    "save_path": "results/attacks/distillation/" + method_wat + "/" + cf_mlp_riga_dict[
        "architecture"].lower() + "/" + save_path_attack + ".pth",
    "device":cf_mlp_riga_dict["device"],
    "attack_type":attack_type,
    "layer_name": layer_name,
    "epoch_attack": epoch_attack

}