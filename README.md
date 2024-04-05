# DICTION: DynamIC robusT whIte bOx watermarkiNg scheme for Deep Neural Networks

Deep Neural Network (DNN) watermarking is a potent method for asserting the ownership of deep learning models. These models often result from computationally intensive processes and meticulously compiled and annotated datasets. By embedding a secret identifier (watermark) within the model, an owner can retrieve it to prove ownership. Among the pioneering techniques, DeepSigns offers an effective dynamic white-box watermarking approach, preserving the accuracy of the prediction model without altering the statistical distribution of model activation maps. However, DeepSigns watermarked models become vulnerable to multiple attacks when incorporating large watermarks, limiting the number of message bits that can be inserted for maintaining robustness. This paper introduces a unified framework to formalize white-box watermarking schemes and proposes a novel dynamic white-box watermarking scheme, "DICTION," which extends "DeepSigns." Its uniqueness stems from employing adversarial learning with data generated from a latent space. Experimental results on the same test set used by DeepSigns show that DICTION achieves superior capacity without compromising accuracy or robustness.

The paper preprint is available at <http://arxiv.org/abs/2210.15745>.

## Overview

The DICTION project is developed in Python and encompasses a variety of operations associated with different methods and models in the realm of watermarking. These operations include watermarking, training, fine-tuning, overwriting, pruning, showing, and Property Inference Attack (PIA).

## Features

We have configured all watermarking schemes and removal attacks for the image classification datasets: CIFAR-10 (32x32 pixels, 10 classes) and MNIST (28x28 pixels, 10 classes). The implemented watermarking schemes are:

1. [DeepSigns](https://www.microsoft.com/en-us/research/uploads/prod/2018/11/2019ASPLOS_Final_DeepSigns.pdf)
2. [Uchida](https://dl.acm.org/doi/10.1145/3078971.3078974)
3. [Encryption Resistant Scheme](https://ieeexplore.ieee.org/document/9746461)
4. [RIGA](https://dl.acm.org/doi/10.1145/3442381.3450000)
5. [DICTION](https://arxiv.org/abs/2210.15745)

Implemented attacks include fine-tuning, pruning, overwriting, and property inference attacks, allowing for extensive experimentation.

## Requirements

To set up the project environment, you'll need:

- Python>=3.10
- torch~=1.13.1
- numpy~=1.23.5
- scipy~=1.10.0
- matplotlib~=3.5.3
- torchvision~=0.14.1
- tqdm~=4.65.0
- Pillow~=9.4.0

## Setup

1. Clone the repository.
2. Navigate to the project directory.
3. Create a virtual environment:

    ```bash
    python3 -m venv venv
    ```

4. Activate the virtual environment:

    ```bash
    source venv/bin/activate
    ```

5. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

## Running Tests

Use the `run_tests.sh` script to perform various tests. It activates the virtual environment and runs the `test_case.py` script with different combinations of methods, models, and operations. Available methods include DICTION, DEEPSIGNS, UCHIDA, RES_ENCRYPT, and RIGA. Models include MLP, CNN, RESNET18, and MLP_RIGA, with operations such as WATERMARKING, TRAINING, FINE_TUNING, OVERWRITING, PRUNING, SHOW, and PIA. Execute the tests with:

```bash
bash ./run_tests.sh
```

This will run the `test_case.py` script for each combination of method, model, and operation, and output the results to `out.txt`.

## Configuration Files

This project includes several configuration files that allow you to customize various aspects of the data, training, and watermarking processes.

- `cf_data.py`: This file contains configurations related to data such as batch size and the path to save it. For example:

    ```python
    config_data = {
        "database": "my_database", # CIFAR-10 or MNIST
        "batch_size": 128, # batch size
        "device": "cuda", # device
        "path_to_save_data": "/data/my_data" # path to save data 
    }
    ```

- `cf_train` directory: This directory contains configuration files for training different models. For example, `cf_mlp.py` includes all the hyperparameters for training the MLP model and the path to save the trained model. For example:

    ```python
    config_model = {
        "lr": 0.001,  # learning rate
        "epochs": 50, # number of epochs
        "wd": 0, # weight decay
        "opt": "Adam", # optimizer
        "batch_size": config_data["batch_size"], # batch size
        "architecture": "MLP", # model architecture
        "milestones": [25, 45], # milestones
        "gamma": 0.1, # gamma related to the scheduler
        "criterion": nn.CrossEntropyLoss(), # loss function
        "scheduler": "MultiStepLR", # scheduler
        "device": config_data["device"], # device
        "database": config_data["database"], # database
        "momentum": 0, # momentum for the optimizer 
        "save_path": "/models/my_model" # path to save the model
    }
    ```

- `cf_watermark` directory: This directory contains configuration files for different watermarking schemes. For example, `cf_diction.py` includes the watermarking parameters for each model to watermark, such as the size of the watermark, the layer to watermark, the number of epochs to embed the watermark, and so on. For example:

    ```python
    cf_mlp_embed = {
        "configuration": config_model, # model configuration
        "database": config_model["database"], # database
        "watermark_size": 512, # watermark size
        "epochs": 10000, # the maximum number of epochs to embed the watermark
        "epoch_check": 30, # the number of epochs to check the watermark, if ber is 0, the watermark is embedded, and the process stops
        "batch_size": int(config_model["batch_size"] * 0.5), #  batch size of the trigger set, e.g. half size of the batch size of the training set    
        "mean": 0, # mean to generate the trigger set 
        "std": 1, # std to generate the trigger set
        # To customize the trigger set, you can add a square into the images of the trigger set, the following parameters are used to generate the square
        "square_size": 5, # square size to add it into the trigger set
        "start_x": 0, # start x to generate the square     
        "start_y": 0, # start y to generate the square
        "square_value": 1, # value of the square
    
        "n_features": 0.5, #  the percentage of features to be used in the watermarking process
        "lambda": 1e-0, # the regularization parameter to train the projection model
        "layer_name": "fc2",  # the layer name to be watermarked
        "lr": 1e-3, # learning rate to embed the watermark into the target model
        "wd": 0, # weight decay to embed the watermark into the target model
        "opt": "Adam", # optimizer to embed the watermark into the target model
        "scheduler": "MultiStepLR", # scheduler to embed the watermark into the target model
        "architecture": config_model['architecture'], # architecture of the model to watermark
        "momentum": 0, # momentum to embed the watermark into the target model
        "milestones": [20, 3000], # milestones to embed the watermark into the target model
        "gamma": .1,    # gamma to embed the watermark into the target model
        "criterion": config_model["criterion"], # loss function to embed the watermark into the target model
        "device": config_model["device"],   # device to embed the watermark into the target model
        "path_model": config_model["save_path"], # path to load the original model
        "save_path": "results/watermarked_models/diction/" + config_model['architecture'].lower() + "/_lfc2_wat512_ep10000_epc30.pth" # path to save the watermarked model
    }
    ```
  
## Contributing

Contributions are welcome. Please submit a pull request with your changes. If you have any questions, please contact at reda.bellafqira@imt-atlantique.fr

## License

This project is licensed under the terms of the MIT license.

## Cite our paper
```
@article{bellafqira2022diction,
  title={DICTION: DynamIC robusT whIte bOx watermarkiNg scheme},
  author={Bellafqira, Reda and Coatrieux, Gouenou},
  journal={arXiv preprint arXiv:2210.15745},
  year={2022}
}
```
