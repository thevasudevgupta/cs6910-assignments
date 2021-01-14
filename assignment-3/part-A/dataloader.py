
import torch
import pandas as pd


class ContextDataset(torch.utils.data.Dataset):

    def __init__(self, list_of_tuple, word_to_index:dict, skip_gram=True):
        self.data = list_of_tuple
        self.word_to_index = word_to_index

        self.skip_gram = skip_gram

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        w1, w2 = self.data[idx]
        if self.skip_gram:
            return self.word_to_index[w1], self.word_to_index[w2]

        return [self.word_to_index[w] for w in w1], self.word_to_index[w2]


class DataLoader(object):

    def __init__(self, batch_size, vocab_path="../data/vocab.txt", skip_gram=True):
        """
        This class is preparing data for feeding into torch-model
        """
        self.batch_size = batch_size

        with open(vocab_path, "r") as f:
            vocab = f.read().split("\n")
        self.vocab_size = len(vocab)
        self.word_to_index = {w: idx for (idx, w) in enumerate(vocab)}
        self.index_to_word = {idx: w for (idx, w) in enumerate(vocab)}

        self.skip_gram = skip_gram

    def setup(self, list_of_tuple):
        dataset = ContextDataset(list_of_tuple, self.word_to_index, self.skip_gram)
        return dataset

    def train_dataloader(self, dataset):
        return torch.utils.data.DataLoader(dataset, 
                                          batch_size=self.batch_size,
                                          shuffle=True,
                                          collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(features):
        context = [f[0] for f in features]
        target = [f[1] for f in features]
        return torch.tensor(context), torch.tensor(target)
