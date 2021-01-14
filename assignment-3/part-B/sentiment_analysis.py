# __name__ = "Vasudev Gupta"
"""
    This is main script to start training model for classification task into 5 categories

    USAGE:
        `python sentiment_analysis.py [--options]`
"""

import torch.nn as nn
import torch
import torch.nn.functional as F

import wandb

from dataloader import DataLoader
from trainer import Trainer
from utils import Classifier, get_parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()
    wandb.init(config=args, project="CS6910-assignment3", name=args.wandb_run_name)

    dl = DataLoader(args.batch_size, args.max_length, args.vocab_path)
    tr_data, val_data, test_data = dl.setup()
    tr_data = dl.train_dataloader(tr_data)
    val_data = dl.val_dataloader(val_data)
    test_data = dl.test_dataloader(test_data)

    classifier = Classifier(dl.vocab_size, args.embedding_size, args.num_lstm_layers, args.dropout)

    trainer = Trainer(classifier, args)
    trainer.fit(tr_data, val_data, test_data)
