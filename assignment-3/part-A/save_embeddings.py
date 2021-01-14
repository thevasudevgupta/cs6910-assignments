# __author__ = "Vasudev Gupta"
"""
    This script will extract & save embedding weights into the `data` directory
    from the weights trained by `word2vec.py` script in this directory

    USAGE:
        python save_embedding.py [--options]
"""

import argparse
import torch

from utils import Word2Vec
from dataloader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--embedding_size", type=int, default=300)
parser.add_argument("--weights_path", type=str, default="weights/cbow-300d-2w.pt")
parser.add_argument("--lstm_based", action="store_true")

if __name__ == "__main__":

    args = parser.parse_args()
    dl = DataLoader(batch_size=None)
    state_dict = torch.load(args.weights_path, map_location=torch.device("cpu"))
    model = Word2Vec(args.embedding_size, dl.vocab_size, lstm_based=args.lstm_based)
    model.load_state_dict(state_dict)

    save_path = args.weights_path.split("/")[-1]
    torch.save(model.embed.weight.data, f"../data/embed-{save_path}")
    print("embedding saved")
