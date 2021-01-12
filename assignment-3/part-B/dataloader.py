import torch
import pandas as pd

class ClassificationDataset(torch.utils.data.Dataset):

    def __init__(self, file_name:str):
        self.data = pd.read_table(file_name)        

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        return {
            "text": self.data.loc[idx, "Phrase"],
            "sentiment": self.data.loc[idx, "Sentiment"]
        }


class DataLoader(object):

    def __init__(self, batch_size):
        """
        This class is preparing data for feeding into torch-model
        """
        self.batch_size = batch_size

    def setup(self):
        self.tr_data = ClassificationDataset("../data/train.csv")
        self.val_data = ClassificationDataset("../data/val.csv")
        self.test_data = ClassificationDataset("../data/test.csv")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.tr_data, batch_size=self.batch_size,
                                          shuffle=True, num_workers=4,
                                          pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size,
                                         shuffle=False, num_workers=4,
                                         pin_memory=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size,
                                         shuffle=False, num_workers=4,
                                         pin_memory=True)
