
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


def get_cosine_similarity(word1, word2, model:nn.Module, weights_path:str, vocab_path="../data/vocab.txt"):

    with open(vocab_path, "r") as f:
        vocab = f.read().split("\n")
    word_to_index = {w: idx for (idx, w) in enumerate(vocab)}
    index_to_word = {idx: w for (idx, w) in enumerate(vocab)}

    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")

    wts = torch.load(weights_path, map_location=torch.device("cpu"))
    model.load_state_dict(wts)

    e1 = model.embed(torch.tensor(word_to_index[word1])).view(1, -1).to(device)
    e2 = model.embed(torch.tensor(word_to_index[word2])).view(1, -1).to(device)
    cs = torch.nn.CosineSimilarity(-1).to(device)

    return cs(e1, e2).item()
