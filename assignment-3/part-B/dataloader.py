import torch
import pandas as pd


class ClassificationDataset(torch.utils.data.Dataset):

    def __init__(self, file_name:str, word_to_index:dict):
        self.data = pd.read_table(file_name)
        self.word_to_index = word_to_index        
        self.unk = word_to_index["<UNK>"]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        w1 = self.data.loc[idx, "Phrase"].lower()
        w2 = self.data.loc[idx, "Sentiment"]
        return {
            "text": [self.word_to_index.get(w, self.unk) for w in w1.split(" ")],
            "sentiment": w2
        }


class DataLoader(object):

    def __init__(self, batch_size, max_length, vocab_path):
        """
        This class is preparing data for feeding into torch-model
        """
        self.batch_size = batch_size
        self.max_length = max_length

        with open(vocab_path, "r") as f:
            vocab = f.read().split("\n")
        self.vocab_size = len(vocab)
        self.word_to_index = {w: idx for (idx, w) in enumerate(vocab)}
        self.index_to_word = {idx: w for (idx, w) in enumerate(vocab)}

        self.word_to_index["<UNK>"] = self.vocab_size
        self.word_to_index["<PAD>"] = self.vocab_size + 1

        self.index_to_word[self.vocab_size] = "<UNK>"
        self.index_to_word[self.vocab_size+1] = "<PAD>"

    def setup(self):
        tr_data = ClassificationDataset("../data/train.csv", self.word_to_index)
        val_data = ClassificationDataset("../data/val.csv", self.word_to_index)
        test_data = ClassificationDataset("../data/test.csv", self.word_to_index)
        print("datasets lengths : ", len(tr_data), len(val_data), len(test_data))
        return tr_data, val_data, test_data

    def train_dataloader(self, train_data):
        return torch.utils.data.DataLoader(train_data, batch_size=self.batch_size,
                                          shuffle=True, num_workers=4,
                                          pin_memory=True, collate_fn=self.collate_fn)

    def val_dataloader(self, val_data):
        return torch.utils.data.DataLoader(val_data, batch_size=self.batch_size,
                                         shuffle=False, num_workers=4,
                                         pin_memory=True, collate_fn=self.collate_fn)

    def test_dataloader(self, test_data):
        return torch.utils.data.DataLoader(test_data, batch_size=self.batch_size,
                                         shuffle=False, num_workers=4,
                                         pin_memory=True, collate_fn=self.collate_fn)

    def collate_fn(self, features):
        text = [t["text"] for t in features]
        sentiment = [t["sentiment"] for t in features]

        max_length = max([len(l) for l in text])
        text = torch.tensor([self._pad(t, max_length) for t in text])

        if self.max_length > 0:
            text = text[:, :self.max_length]

        return {
          "text": text,
          "sentiment": torch.tensor(sentiment)
        }

    def _pad(self, ls:list, max_length:int):
        while len(ls) < max_length:
            ls.append(self.word_to_index["<PAD>"])
        return ls     
