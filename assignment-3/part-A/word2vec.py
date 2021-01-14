# __author__ = "Vasudev Gupta"
"""
    This is the main the script that will enable training ...

    USAGE:
        python word2vec.py [--options]
        
        `training_id` option will allow you to switch among `bag of words` or `skip gram` or `lstm based model`
"""

import argparse
from dataclasses import dataclass
import wandb

from dataloader import DataLoader
from trainer import Trainer
from utils import Word2Vec, prepare_data

parser = argparse.ArgumentParser()
parser.add_argument("--embedding_size", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=1600)
parser.add_argument("--lr", type=float, default=5.e-3)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--save_path", type=str, default="wts")
parser.add_argument("--wandb_run_name", type=str, default=None)
parser.add_argument("--window_size", type=int, default=2)
parser.add_argument("--training_id", type=str, default="skip_gram", help='either of "cbow" or "skip_gram" or "lstm_based"')


if __name__ == "__main__":

    args = parser.parse_args()
    wandb.init(config=args, project="CS6910-assignment3", name=args.wandb_run_name)

    with open("../data/text8.txt", "r") as f:
        corpus = f.read().split(" ")

    context = prepare_data(corpus, window_size=args.window_size)

    dl = DataLoader(args.batch_size)
    context = dl.setup(context)
    context = dl.train_dataloader(context)

    model = Word2Vec(args.embedding_size, dl.vocab_size, lstm_based=False)
    if args.training_id=="lstm_based":
        model = Word2Vec(args.embedding_size, dl.vocab_size, lstm_based=True)

    trainer = Trainer(model, args)
    trainer.fit(context)
