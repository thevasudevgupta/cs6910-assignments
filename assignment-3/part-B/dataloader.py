import torch
import pandas as pd


class ClassificationDataset(torch.utils.data.Dataset):

    def __init__(self, file_name:str, word_to_index:dict):
        self.data = pd.read_table(file_name)
        self.word_to_index = word_to_index        

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        w1 = self.data.loc[idx, "Phrase"]
        w2 = self.data.loc[idx, "Sentiment"]
        return {
            "text": [self.word_to_index[w] for w in w1.split(" ")],
            "sentiment": w2
        }


class DataLoader(object):

    def __init__(self, batch_size, vocab_path):
        """
        This class is preparing data for feeding into torch-model
        """
        self.batch_size = batch_size

        with open(vocab_path, "r") as f:
            vocab = f.read().split("\n")
        self.vocab_size = len(vocab)
        self.word_to_index = {w: idx for (idx, w) in enumerate(vocab)}
        self.index_to_word = {idx: w for (idx, w) in enumerate(vocab)}

    def setup(self):
        tr_data = ClassificationDataset("../data/train.csv", self.word_to_index)
        val_data = ClassificationDataset("../data/val.csv", self.word_to_index)
        test_data = ClassificationDataset("../data/test.csv", self.word_to_index)
        return tr_data, val_data, test_data

    def train_dataloader(self, train_data):
        return torch.utils.data.DataLoader(train_data, batch_size=self.batch_size,
                                          shuffle=True, num_workers=4,
                                          pin_memory=True)

    def val_dataloader(self, val_data):
        return torch.utils.data.DataLoader(val_data, batch_size=self.batch_size,
                                         shuffle=False, num_workers=4,
                                         pin_memory=True)

    def test_dataloader(self, test_data):
        return torch.utils.data.DataLoader(test_data, batch_size=self.batch_size,
                                         shuffle=False, num_workers=4,
                                         pin_memory=True)
