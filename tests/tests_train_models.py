import numpy as np
from matplotlib import pyplot as plt
from util.util import Database, TrainModel


class Tests:
    @staticmethod
    def gen_database(config_data):
        """ generate a new database"""
        Database.gen_dataset_loaders(config_data)

    @staticmethod
    def train_model(config_data, config_train):
        """load the database and train the model"""
        """Load data"""
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        """Init the model"""
        init_model = TrainModel.get_model(config_train["architecture"], config_train["device"])
        print("Model to train...")
        print(init_model)
        """Start training the model"""
        if config_train["show_acc_epoch"]:
            _, acc_list = TrainModel.fine_tune(init_model, train_loader, test_loader, config_train)
            Tests.plot_acc(acc_list, config_train)
        else:
            TrainModel.fine_tune(init_model, train_loader, test_loader, config_train)

    @staticmethod
    def plot_acc(acc_list, config_train):
        epochs = np.arange(len(acc_list))
        plt.plot(epochs, acc_list, c="black", marker='*', label=f"ACC of {config_train['architecture']} "
                                                                f"over database = {config_train['database']}")
        # for a, acc in zip(epochs, acc_list):
        #     plt.text(a + 0.02, acc + 0.02, str(acc))
        plt.xticks(np.arange(-1, len(acc_list) + 2, 1))
        plt.yticks(np.arange(-10, 110, 10))
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()
        plt.grid(True)
        plt.title(f"ACC of {config_train['architecture']} over database = {config_train['database']}")
        plt.savefig(config_train['save_fig_path'])
        plt.show()

