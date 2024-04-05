import os
from copy import deepcopy
from argparse import ArgumentParser
import datetime

# --------------- Data configuration ---------------
from configs.cf_data.cf_data import cf_mnist_data, cf_cifar10_data

# --------------- Models configuration ---------------
from configs.cf_train.cf_cnn import cf_cnn_dict
from configs.cf_train.cf_mlp import cf_mlp_dict
from configs.cf_train.cf_resnet18 import cf_resnet18_dict
from configs.cf_train.cf_mlp_riga import cf_mlp_riga_dict

# --------------- Default tests configuration ---------------
from tests.tests_common import Tests

# Specify test scenario
parser = ArgumentParser()
parser.add_argument("--method", type=str, default="DICTION")
parser.add_argument("--model", type=str, default="MLP")
parser.add_argument("--operation", type=str, default="TRAIN")
args = parser.parse_args()
method = args.method
model = args.model
operation = args.operation

#  ---------------Watermarking configurations ---------------
match method:
    case 'DEEPSIGNS':
        from configs.cf_watermark.cf_deepsigns import (cf_cnn_embed, cf_cnn_attack_ft, cf_resnet18_embed, \
                                                       cf_resnet18_attack_ft, cf_mlp_embed, cf_mlp_attack_ft,
                                                       cf_mlp_attack_pr, cf_mlp_attack_ow,
                                                       cf_cnn_attack_ow, \
                                                       cf_cnn_attack_pr, cf_resnet18_attack_pr, cf_resnet18_attack_ow,
                                                       cf_mlp_riga_embed, cf_mlp_riga_attack_ft, \
                                                       cf_mlp_riga_attack_pr, cf_mlp_riga_attack_ow,
                                                       cf_mlp_riga_attack_pia, cf_mlp_attack_pia, cf_cnn_attack_pia, \
                                                       cf_resnet18_attack_pia)

    case 'DICTION':
        from configs.cf_watermark.cf_diction import cf_cnn_embed, cf_cnn_attack_ft, cf_resnet18_embed, \
            cf_resnet18_attack_ft, cf_mlp_embed, cf_mlp_attack_ft, cf_mlp_attack_pr, cf_mlp_attack_ow, \
            cf_mlp_attack_pia, \
            cf_cnn_attack_ow, cf_cnn_attack_pr, cf_resnet18_attack_pr, cf_resnet18_attack_ow, cf_cnn_attack_pia, \
            cf_resnet18_attack_pia, cf_mlp_riga_embed, cf_mlp_riga_attack_ft, cf_mlp_riga_attack_pr, \
            cf_mlp_riga_attack_ow, \
            cf_mlp_riga_attack_pia

    case 'UCHIDA':
        from configs.cf_watermark.cf_uchida import (cf_mlp_embed, cf_cnn_embed, cf_mlp_attack_ft, cf_mlp_attack_pr,
                                                    cf_mlp_attack_ow, cf_mlp_attack_pia)

    case 'RES_ENCRYPT':
        from configs.cf_watermark.cf_res_encrypt import cf_cnn_embed, cf_cnn_attack_ft, cf_resnet18_embed, \
            cf_resnet18_attack_ft, cf_mlp_embed, cf_mlp_attack_ft, cf_mlp_attack_pr, cf_mlp_attack_ow, \
            cf_mlp_attack_pia, \
            cf_cnn_attack_ow, cf_cnn_attack_pr, cf_resnet18_attack_pr, cf_resnet18_attack_ow, cf_cnn_attack_pia, \
            cf_resnet18_attack_pia, cf_mlp_riga_embed, cf_mlp_riga_attack_ft, cf_mlp_riga_attack_pr, \
            cf_mlp_riga_attack_ow, \
            cf_mlp_riga_attack_pia
    case 'RIGA':
        from configs.cf_watermark.cf_riga import cf_mlp_embed
    case _:
        raise ValueError(f"Method {method} not found")

# --------------- Model configurations ---------------
match model:
    case 'MLP':
        config_train, config_data, config_embed = cf_mlp_dict, cf_mnist_data, cf_mlp_embed  # Training and data
        config_attack_ft, config_attack_pr, config_attack_ow, config_attack_pia = \
            cf_mlp_attack_ft, cf_mlp_attack_pr, cf_mlp_attack_ow, cf_mlp_attack_pia  # Attacks
    case 'CNN':
        config_train, config_data, config_embed = cf_cnn_dict, cf_cifar10_data, cf_cnn_embed
        config_attack_ft, config_attack_pr, config_attack_ow, config_attack_pia = \
            cf_cnn_attack_ft, cf_cnn_attack_pr, cf_cnn_attack_ow, cf_cnn_attack_pia
    case 'RESNET18':
        config_train, config_data, config_embed = cf_resnet18_dict, cf_cifar10_data, cf_resnet18_embed
        config_attack_ft, config_attack_pr, config_attack_ow, config_attack_pia = \
            cf_resnet18_attack_ft, cf_resnet18_attack_pr, cf_resnet18_attack_ow, cf_resnet18_attack_pia
    case 'MLP_RIGA':
        config_train, config_data, config_embed = cf_mlp_riga_dict, cf_mnist_data, cf_mlp_riga_embed
        config_attack_ft, config_attack_pr, config_attack_ow, config_attack_pia = \
            cf_mlp_riga_attack_ft, cf_mlp_riga_attack_pr, cf_mlp_riga_attack_ow, cf_mlp_riga_attack_pia
    case _:
        raise ValueError(f"Model {model} not found")

#  --------------- Tests configuration ---------------
_tests = Tests(method, model)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#  --------------- Main  ---------------
if __name__ == '__main__':
    print(f"\n#####################Running {method} with {model}#############################\n")

    # ######################### Data generation ##########################
    if operation == "TRAIN":
        t_start_gen_db = datetime.datetime.now()
        Tests.gen_database(config_data)
        runtime_gen_db = datetime.datetime.now() - t_start_gen_db
        # ######################### Train networks ##########################
        print("### Start the model training ### ")
        t_start_train = datetime.datetime.now()
        Tests.train_model(config_data, config_train)
        runtime_train = datetime.datetime.now() - t_start_train

    # ######################### run embedding ##########################
    if operation == "WATERMARKING":
        print("\n---------------------- Watermarking -----------------------------\n")
        # ### run GMM for DeepSigns ###
        if method == "DEEPSIGNS":
            Tests.gmm(config_embed, config_data)

        acc = 0
        ber = 0
        nb_run = 1
        for i in range(nb_run):
            t_start_wm = datetime.datetime.now()
            temp_acc, temp_ber = _tests.embedding(deepcopy(config_embed), deepcopy(config_data), i + 1)
            runtime_wm = datetime.datetime.now() - t_start_wm
            acc = acc + temp_acc
            ber = ber + temp_ber
        print("acc=", acc / nb_run)
        print("ber=", ber / nb_run)

        # Compute the average runtime for the watermarking process and save it in a file

        # with open("runtimes.txt", "a+") as f :
        #     f.write(f"\nRuntimes for {model}\n")
        #     f.write(f"Database generation : {runtime_gen_db}\
        #             \nTraining model : {runtime_train}\
        #             \nWatermarking with {method} : {runtime_wm}")

    # ######################### run fine-tuning attack ##########################
    elif operation == "FINE_TUNING":
        print("\n---------------------- Fine-tuning attack -----------------------------\n")
        _tests.fine_tune_attack(config_embed, config_attack_ft, config_data)

    # ######################### run pruning attack ##########################
    elif operation == "PRUNING":
        print("\n---------------------- Pruning attack ---------------------------------\n")
        _tests.pruning_attack(config_embed, config_attack_pr, config_data)

    # ######################### run Overwriting attack ##########################
    elif operation == "OVERWRITING":
        # ### run GMM for DeepSigns ###
        if method == "DEEPSIGNS":
            _tests.gmm(config_attack_ow, config_data)
        print("\n---------------------- Overwriting attack -----------------------------\n")
        _tests.overwriting_attack(config_embed, config_attack_ow, config_data)

    # ######################### run PIA attack ##########################
    elif operation == "PIA":
        print("\n---------------------- PIA attack -----------------------------\n")
        _tests.pia_attack(config_data, config_embed, config_attack_pia)

    # ######################### run show Weights and activations distribution ##########################
    elif operation == "SHOW":
        print("\n---------------------- Weights and activations distribution -----------------------------\n")
        _tests.show_weights_distribution(config_embed, config_data)
