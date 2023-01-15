from util.util import TrainModel, Database
from watermark import uchida


class Tests:

    @staticmethod
    def embedding(config_embed, config_data, nb_run=0):
        """embedding """
        # Get model
        init_model = TrainModel.get_model(config_embed["architecture"], config_embed["device"])
        init_model.load_state_dict(TrainModel.load_model(config_embed["path_model"])['net'])
        # Get data
        train_loader, test_loader = Database.load_dataset_loaders(config_data)
        # evaluate the model
        TrainModel.evaluate(init_model, test_loader, config_embed)

        # embed the watermark with Uchida
        model_wat, ber = uchida.embed(init_model, test_loader, train_loader, config_embed)

        acc = TrainModel.evaluate(model_wat, test_loader, config_embed)

        return acc, ber
