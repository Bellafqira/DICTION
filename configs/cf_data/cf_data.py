import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ******************* CF Mnist *******************
database = "mnist"
batch_size = 512

cf_mnist_data = {"batch_size": batch_size, "database": database, "device": device,
                 "show_acc_epoch": True, "save_path": f"datasets\_db{database}\_bs{batch_size}"}

# ******************* CF Cifar10 *******************
database = "cifar10"
batch_size = 128

cf_cifar10_data = {"batch_size": batch_size, "database": database,
                   "show_acc_epoch": False, "save_path": f"datasets\_db{database}\_bs{batch_size}"}
