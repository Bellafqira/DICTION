import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ******************* CF Mnist *******************
database = "mnist"
batch_size = 128

cf_mnist_data = {"batch_size": batch_size, "database": database, "device": device,
                 "save_path": f"datasets/{database}/_db{database}_bs{batch_size}.pth"}

# ******************* CF Cifar10 *******************
database = "cifar10"
batch_size = 128

cf_cifar10_data = {"batch_size": batch_size, "database": database, "device": device,
                   "save_path": f"datasets/{database}/_db{database}_bs{batch_size}.pth"}
