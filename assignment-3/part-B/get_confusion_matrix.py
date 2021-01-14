# __author__ = "Vasudev Gupta"
"""
    This will print the confusion matrix for you

    USAGE:
        `python get_confusion_matrix.py [--options]`
"""

from sklearn.metrics import confusion_matrix
import pandas as pd
import torch
import argparse

from utils import Classifier
from dataloader import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--weights_path", type=str)
parser.add_argument("--embedding_size", type=int)
parser.add_argument("--num_lstm_layers", type=int, default=2)
parser.add_argument("--dropout_prob", type=float, default=0.1)

if __name__ == '__main__':

    args = parser.parse_args()

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    dl = DataLoader(batch_size=32)
    test_data = dl.setup(return_only_test_data=True)
    test_data = dl.test_dataloader(test_data)

    wts = torch.load(args.weights_path, map_location=torch.device("cpu"))
    model = Classifier(dl.vocab_size, args.embedding_size, args.num_lstm_layers, args.dropout_prob)
    model.load_state_dict(wts)
    model.to(device)

    preds, labels = model.predict(test_data, device)

    cm = confusion_matrix(labels, preds)
    names = ["negative", "somewhat negative", "neutral", "somewhat positive", "positive"]
    cm = pd.DataFrame(cm, columns=names, index=names)
    print(cm)
