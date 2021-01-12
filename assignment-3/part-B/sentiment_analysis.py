import torch.nn as nn
import torch
import torch.nn.functional as F

from dataclasses import dataclass
import wandb

from dataloader import DataLoader
from trainer import Trainer

class Classifier(nn.Module):

    def __init__(self, hidden_size=256, num_lstm_layers=2, dropout=0.1):
        self.embedding = nn.Embedding()
        self.lstm = nn.LSTM(300, hidden_size, num_layers=num_lstm_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, 5)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        return self.linear(x)

@dataclass
class args:
    hidden_size = 300
    num_lstm_layers = 2
    dropout = 0.1
    batch_size = 32
    lr = 1.e-3
    weight_decay = 0.001
    epochs = 2
    save_path = "partB-wts"
    wandb_run_name = "run"

if __name__ == "__main__":

    wandb.init(config=args.__dict__, project="CS6910-assignment1", name=args.wandb_run_name)

    dl = DataLoader(args.batch_size)
    dl.setup()
    tr_dataset = dl.train_dataloader()
    val_dataset = dl.val_dataloader()
    tst_dataset = dl.test_dataloader()

    classifier = Classifier(args.hidden_size, args.num_lstm_layers, args.dropout)

    trainer = Trainer(classifier, args)
    trainer.fit(tr_dataset, val_dataset, tst_dataset)
