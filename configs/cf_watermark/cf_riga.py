from configs.cf_train.cf_cnn import cf_cnn_dict
from configs.cf_train.cf_mlp import cf_mlp_dict

# ************* watermarking MLP architecture ****************
watermark_size = 256
epochs_embed = 1000
layer_name = "fc1.weight"
epoch_check = 10
save_path_embed = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epochs_embed) +\
                  "_epc" + str(epoch_check)

cf_mlp_embed = {"configuration": cf_mlp_dict,
                "database": cf_mlp_dict["database"],
                "watermark_size": watermark_size,
                "epochs": epochs_embed,
                "epoch_check": epoch_check,

                "batch_size": 128,
                "lambda_1": 0.01,
                "lambda_2": 0.1,

                "layer_name": layer_name,
                "lr": 1e-4,
                "wd": 0,
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

# ************* watermarking CNN architecture ****************
watermark_size = 256
epochs_embed = 10000
layer_name = "fc1"
epoch_check = 10
save_path_embed = "_l" + layer_name + "_wat" + str(watermark_size) + "_ep" + str(epochs_embed) + "_epc" + \
                  str(epoch_check)

cf_cnn_embed = {"configuration": cf_cnn_dict,
                "database": cf_cnn_dict["database"],
                "watermark_size": watermark_size,
                "epochs": epochs_embed,
                "epoch_check": epoch_check,

                # Latent space parameters
                "batch_size": cf_cnn_dict["batch_size"] // 2,
                "mean": 0,
                "std": 1,
                "n_features": 512 // 1,
                "n_features_layer": 512,
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