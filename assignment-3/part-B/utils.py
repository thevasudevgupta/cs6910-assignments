
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

class Classifier(nn.Module):

    def __init__(self, vocab_size, embedding_size=256, num_lstm_layers=2, dropout=0.1, embedding_path=None):
        super().__init__()
        vocab_size += 2 # handling unknown & pad tokens
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        if embedding_path:
            embed = torch.load(embedding_path)
            self.embedding.weight.data = embed
        self.lstm = nn.LSTM(embedding_size, embedding_size, num_layers=num_lstm_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(embedding_size, 5)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.linear(x)
    
    @torch.no_grad()
    def predict(self, dataset, device=torch.device("cpu")):
        preds = []
        labels = []

        for batch in tqdm(dataset, desc="predicting .. "):
            inputs = batch["text"].to(device)
            out = self(inputs)
            _, pred = out.max(1)
            preds.extend(pred.cpu().tolist())
            labels.extend(batch["sentiment"].tolist())

        return preds, labels


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_size", type=int, default=300)
    parser.add_argument("--embedding_path", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=-1)
    parser.add_argument("--vocab_path", type=str, default="../data/vocab.txt")
    parser.add_argument("--num_lstm_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save_path", type=str, default="wts.pt")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    return parser

