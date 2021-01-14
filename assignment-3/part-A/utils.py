
from tqdm import tqdm
import torch.nn  as  nn
import torch


class Word2Vec(nn.Module):

    def __init__(self, embedding_size, vocab_size, lstm_based=False, cbow=False):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)

        self.lstm_based = lstm_based
        self.cbow = cbow

        if self.lstm_based:
            self.lstm = nn.LSTM(embedding_size, embedding_size, batch_first=True)

    def forward(self, context_word, target=None):

        x = self.embed(context_word)

        if self.cbow:
            x = x.sum(dim=1)
        elif self.lstm_based and (target is not None):
            target = self.embed(target).unsqueeze(1)
            x = torch.cat([x, target], dim=1)
            x, _ = self.lstm(x)
            x = x[:, -1, :]

        x = self.linear(x)

        return x


def prepare_data(corpus, window_size=4, skip_gram=True):
    context = []
    w = window_size
    if skip_gram:
        for i, word in tqdm(enumerate(corpus), total=len(corpus), desc=""):
            first_idx = max(0,i-w)
            last_idx = min(i+w, len(corpus))
            for j in range(first_idx, last_idx):
                if i!=j:
                    context.append((word, corpus[j]))
    else:
        for i, word in tqdm(enumerate(corpus), total=len(corpus), desc=""):
            first_idx = max(0,i-w)
            last_idx = min(i+w+1, len(corpus))
            c = []
            for j in range(first_idx, last_idx):
                if i!=j:
                    c.append(corpus[j])
            if len(c) == 2*w:
                context.append((c, word))
    print("Total num of samples : {}".format(len(context)))
    return context


def get_cosine_similarity(word1, word2, model:nn.Module, weights_path:str, vocab_path="../data/vocab.txt"):

    with open(vocab_path, "r") as f:
        vocab = f.read().split("\n")
    word_to_index = {w: idx for (idx, w) in enumerate(vocab)}
    # index_to_word = {idx: w for (idx, w) in enumerate(vocab)}

    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")

    wts = torch.load(weights_path, map_location=torch.device("cpu"))
    model.load_state_dict(wts)

    e1 = model.embed(torch.tensor(word_to_index[word1])).view(1, -1).to(device)
    e2 = model.embed(torch.tensor(word_to_index[word2])).view(1, -1).to(device)
    cs = torch.nn.CosineSimilarity(-1).to(device)

    return cs(e1, e2).item()
