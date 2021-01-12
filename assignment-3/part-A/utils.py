
from tqdm import tqdm
import torch.nn  as  nn
import torch


class Word2Vec(nn.Module):

    def __init__(self, embedding_size, vocab_size, lstm_based=False):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)
        self.lstm_based = lstm_based
        if self.lstm_based:
            self.lstm = nn.LSTM(embedding_size, embedding_size, batch_first=True)

    def forward(self, context_word, **kwargs):
        x = self.embed(context_word)
        if self.lstm_based:
            assert "target" in kwargs
            x = x.unsqueeze(1)
            target = self.embed(kwargs["target"]).unsqueeze(1)
            x = torch.cat([x, target], dim=1)
            x, _ = self.lstm(x)
            x = x.squeeze()[:, -1]
        x = self.linear(x)
        return x


def prepare_data(corpus, window_size=4):
    context = []
    w = window_size
    for i, word in tqdm(enumerate(corpus), total=len(corpus), desc=""):
        first_idx = max(0,i-w)
        last_idx = min(i+w, len(corpus))
        for j in range(first_idx, last_idx):
            if i!=j:
                context.append((corpus[j], word))
    print("Total num of samples : {}".format(len(context)))
    return context