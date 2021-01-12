
import argparse
from dataclasses import dataclass
import wandb

from dataloader import DataLoader
from trainer import Trainer
from utils import Word2Vec, prepare_data

parser = argparse.ArgumentParser()
parser.add_argument("--embedding_size", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=2000)
parser.add_argument("--lr", type=float, default=1.e-3)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--save_path", type=str, default="wts.pt")
parser.add_argument("--wandb_run_name", type=str, default=None)
parser.add_argument("--window_size", type=int, default=4)
parser.add_argument("--training_id", type=str, default="cbow", help='either of "cbow" or "skip_gram" or "lstm_based"')


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
