import os
from copy import deepcopy

# --------------- Data configuration ---------------
from configs.cf_data.cf_data import cf_mnist_data, cf_cifar10_data

#  ---------------Watermarking configurations ---------------
# *********** DeepSigns ***********
# from configs.cf_watermark.cf_deepsigns import cf_cnn_embed, cf_cnn_attack_ft, cf_resnet18_embed, \
#     cf_resnet18_attack_ft, cf_mlp_embed, cf_mlp_attack_ft, cf_mlp_attack_pr, cf_mlp_attack_ow, cf_cnn_attack_ow, \
#     cf_cnn_attack_pr, cf_resnet18_attack_pr, cf_resnet18_attack_ow, cf_mlp_riga_embed, cf_mlp_riga_attack_ft, \
#     cf_mlp_riga_attack_pr, cf_mlp_riga_attack_ow, cf_mlp_riga_attack_pia, cf_mlp_attack_pia, cf_cnn_attack_pia, \
#     cf_resnet18_attack_pia
# from tests.tests_deepsigns import Tests

# *********** DICTION ***********
# from configs.cf_watermark.cf_diction import cf_cnn_embed, cf_cnn_attack_ft, cf_resnet18_embed, \
#     cf_resnet18_attack_ft, cf_mlp_embed, cf_mlp_attack_ft, cf_mlp_attack_pr, cf_mlp_attack_ow, cf_mlp_attack_pia, \
#     cf_cnn_attack_ow, cf_cnn_attack_pr, cf_resnet18_attack_pr, cf_resnet18_attack_ow, cf_cnn_attack_pia, \
#     cf_resnet18_attack_pia, cf_mlp_riga_embed, cf_mlp_riga_attack_ft, cf_mlp_riga_attack_pr, cf_mlp_riga_attack_ow, \
#     cf_mlp_riga_attack_pia
# from tests.tests_diction import Tests

# *********** UCHIDA ***********
# from configs.cf_watermark.cf_uchida import cf_mlp_embed, cf_cnn_embed
# from tests.tests_uchida import Tests

# *********** Enc_Resistant ***********
# from configs.cf_watermark.cf_enc_resistant import cf_mlp_embed, cf_cnn_embed
# from tests.tests_enc_resistant import Tests

# *********** RIGA ***********
from configs.cf_watermark.cf_riga import cf_mlp_embed
from tests.tests_riga import Tests

# --------------- Models configuration ---------------
from configs.cf_train.cf_cnn import cf_cnn_dict
from configs.cf_train.cf_mlp import cf_mlp_dict
from configs.cf_train.cf_resnet18 import cf_resnet18_dict
from configs.cf_train.cf_mlp_riga import cf_mlp_riga_dict


# --------------- Initiate configurations ---------------
# # #             **** MLP data, train and embedding ****
config_train, config_data, config_embed = cf_mlp_dict, cf_mnist_data, cf_mlp_embed
#             **** MLP  watermarking attacks ****
# uncomment later
# config_attack_ft, config_attack_pr, config_attack_ow, config_attack_pia = \
#     cf_mlp_attack_ft, cf_mlp_attack_pr, cf_mlp_attack_ow, cf_mlp_attack_pia

# #             **** CNN data, train and embedding ****
# config_train, config_data, config_embed = cf_cnn_dict, cf_cifar10_data, cf_cnn_embed
# #             **** CNN  watermarking attacks ****
# config_attack_ft, config_attack_pr, config_attack_ow, config_attack_pia =\
#  cf_cnn_attack_ft, cf_cnn_attack_pr, cf_cnn_attack_ow, cf_cnn_attack_pia

# #             **** Resnet18 data, train and embedding ****
# config_train, config_data, config_embed = cf_resnet18_dict, cf_cifar10_data, cf_resnet18_embed
# #             **** Resnet18  watermarking attacks ****
# config_attack_ft, config_attack_pr, config_attack_ow, config_attack_pia =\
#  cf_resnet18_attack_ft, cf_resnet18_attack_pr, cf_resnet18_attack_ow, cf_resnet18_attack_pia

# #             **** MLP_RIGA data, train and embedding ****
# config_train, config_data, config_embed = cf_mlp_riga_dict, cf_mnist_data, cf_mlp_riga_embed
# #             **** MLP_RIGA  watermarking attacks ****
# config_attack_ft, config_attack_pr, config_attack_ow, config_attack_pia = \
#     cf_mlp_riga_attack_ft, cf_mlp_riga_attack_pr, cf_mlp_riga_attack_ow, cf_mlp_riga_attack_pia

# --------------- Tests configuration ---------------
# from tests.tests_train_models import Tests

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#  --------------- Main  ---------------
if __name__ == '__main__':
    # # Train networks
    # Tests.gen_database(config_data)
    # Tests.train_model(config_data, config_train)

    # # Tests on DeepSigns and DICTION
    # # run GMM for DeepSigns
    # Tests.gmm(config_embed, config_data)

    # # run embedding
    acc = 0
    ber = 0
    nb_run = 1
    for i in range(nb_run):
        temp_acc, temp_ber = Tests.embedding(deepcopy(config_embed), deepcopy(config_data), i+1)
        acc = acc + temp_acc
        ber = ber + temp_ber
    print("acc=", acc / nb_run)
    print("ber=", ber / nb_run)

    # # run fine-tuning attack
    # Tests.fine_tune_attack(config_embed, config_attack_ft, config_data)

    # # pruning
    # Tests.pruning_attack(config_embed, config_attack_pr, config_data)

    # # Overwriting
    # run GMM for DeepSigns
    # Tests.gmm(config_attack_ow, config_data)
    # Tests.overwriting_attack(config_embed, config_attack_ow, config_data)

    # # Weights and activations distribution
    # Tests.show_weights_distribution(config_embed, config_data)

    # # PIA Attack
    # Tests.pia_attack(config_data, config_embed, config_attack_pia)
