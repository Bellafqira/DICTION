
from torch.utils.data import Dataset


class CustomTensorDataset(Dataset):
    def __init__(self, x_tensor, y_tensor, transform_list=None):

        self.x_tensor = x_tensor
        self.y_tensor = y_tensor
        self.transforms = transform_list

    def __getitem__(self, index):
        x = self.x_tensor[index]
        y = self.y_tensor[index]
        if self.transforms:
            x = self.transforms(x)
        return x, y

    def __len__(self):
        return len(self.x_tensor)
