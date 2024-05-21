import os
from copy import deepcopy
from argparse import ArgumentParser
import datetime

# Data configuration
from configs.cf_data.cf_data import cf_mnist_data, cf_cifar10_data

# Model configurations
from configs.cf_train.cf_cnn import cf_cnn_dict
from configs.cf_train.cf_mlp import cf_mlp_dict
from configs.cf_train.cf_resnet18 import cf_resnet18_dict
from configs.cf_train.cf_mlp_riga import cf_mlp_riga_dict

# Tests configuration
from tests.tests_common import Tests

# Watermarking method configurations
method_configurations = {
    'DEEPSIGNS': 'configs.cf_watermark.cf_deepsigns',
    'DICTION': 'configs.cf_watermark.cf_diction',
    'UCHIDA': 'configs.cf_watermark.cf_uchida',
    'RES_ENCRYPT': 'configs.cf_watermark.cf_res_encrypt',
    'RIGA': 'configs.cf_watermark.cf_riga',
}

# Model configurations
model_configurations = {
    'MLP': (cf_mlp_dict, cf_mnist_data, 'cf_mlp_embed', 'cf_mlp_attack_ft', 'cf_mlp_attack_pr', 'cf_mlp_attack_ow', 'cf_mlp_attack_pia'),
    'CNN': (cf_cnn_dict, cf_cifar10_data, 'cf_cnn_embed', 'cf_cnn_attack_ft', 'cf_cnn_attack_pr', 'cf_cnn_attack_ow', 'cf_cnn_attack_pia'),
    'RESNET18': (cf_resnet18_dict, cf_cifar10_data, 'cf_resnet18_embed', 'cf_resnet18_attack_ft', 'cf_resnet18_attack_pr', 'cf_resnet18_attack_ow', 'cf_resnet18_attack_pia'),
    'MLP_RIGA': (cf_mlp_riga_dict, cf_mnist_data, 'cf_mlp_riga_embed', 'cf_mlp_riga_attack_ft', 'cf_mlp_riga_attack_pr', 'cf_mlp_riga_attack_ow', 'cf_mlp_riga_attack_pia'),
}

# Parse arguments
parser = ArgumentParser()
parser.add_argument("--method", type=str, default="DICTION")
parser.add_argument("--model", type=str, default="MLP")
parser.add_argument("--operation", type=str, default="TRAIN")
args = parser.parse_args()

method = args.method
model = args.model
operation = args.operation

if method not in method_configurations:
    raise ValueError(f"Method {method} not found")
if model not in model_configurations:
    raise ValueError(f"Model {model} not found")

# Import method-specific configurations
method_module = __import__(method_configurations[method], fromlist=['*'])
globals().update({k: getattr(method_module, k) for k in dir(method_module) if not k.startswith('__')})

# Load model-specific configurations
config_train, config_data, embed_name, ft_name, pr_name, ow_name, pia_name = model_configurations[model]
config_embed = globals()[embed_name]
config_attack_ft = globals()[ft_name]
config_attack_pr = globals()[pr_name]
config_attack_ow = globals()[ow_name]
config_attack_pia = globals()[pia_name]

# Tests configuration
_tests = Tests(method, model)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Main execution
if __name__ == '__main__':
    print(f"\n#####################Running {method} with {model}#############################\n")

    if operation == "TRAIN":
        t_start_gen_db = datetime.datetime.now()
        Tests.gen_database(config_data)
        runtime_gen_db = datetime.datetime.now() - t_start_gen_db
        print(f"runtime_wm to generate the database : {config_data['database']} : {runtime_gen_db} ")

        print("### Start the model training ###")
        t_start_train = datetime.datetime.now()
        Tests.train_model(config_data, config_train)
        runtime_train = datetime.datetime.now() - t_start_train
        print(f"runtime_wm to train the model : {model} : {runtime_train}")

    elif operation == "WATERMARKING":
        print("\n---------------------- Watermarking -----------------------------\n")
        acc, ber, nb_run = 0, 0, 1
        runtime_wm = -1
        for i in range(nb_run):
            t_start_wm = datetime.datetime.now()
            temp_acc, temp_ber = _tests.embedding(deepcopy(config_embed), deepcopy(config_data), i + 1)
            runtime_wm = datetime.datetime.now() - t_start_wm
            acc += temp_acc
            ber += temp_ber

        print("acc=", acc / nb_run)
        print("ber=", ber / nb_run)
        print("runtime_wm to embed the watermark :", runtime_wm)

    elif operation == "FINE_TUNING":
        print("\n---------------------- Fine-tuning attack -----------------------------\n")
        _tests.fine_tune_attack(config_embed, config_attack_ft, config_data)

    elif operation == "PRUNING":
        print("\n---------------------- Pruning attack ---------------------------------\n")
        _tests.pruning_attack(config_embed, config_attack_pr, config_data)

    elif operation == "OVERWRITING":
        print("\n---------------------- Overwriting attack -----------------------------\n")
        _tests.overwriting_attack(config_embed, config_attack_ow, config_data)

    elif operation == "PIA":
        print("\n---------------------- PIA attack -----------------------------\n")
        _tests.pia_attack(config_data, config_embed, config_attack_pia)

    elif operation == "SHOW":
        print("\n---------------------- Weights and activations distribution -----------------------------\n")
        _tests.show_weights_distribution(config_embed, config_data)
