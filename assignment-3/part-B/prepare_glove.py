# __author__ = "Vasudev Gupta"
"""
    This will prepare glove embedding for you ...
    You need to mannually download glove embedding from their official website into `data` directory ...
    ... and then run this script to save them into `.pt` format which will make it easier to use later with torch

    USAGE:
        change variables as per your need and run `python prepare_glove.py`
"""

import torch
from tqdm import tqdm


def get_embedding_weights(file_path, word_to_index, embedding_dim=300):
    """ Loads glove embeddings from a file and
    generates embedding matrix for embedding layer"""

    f = open(file_path, "r")
    embedding_map = {}
    for line in tqdm(f, desc="loading GloVe from local directory"):
        if line == "":
            continue
        word = line.split()[0]
        vec = line.split()[1:]
        embedding_map[word] = torch.tensor([float(v) for v in vec])
    f.close()

    # Generate embedding weights in matrix form
    # Initialize a uniformly distributed tensor of size (vocab_size, embed_dim)
    # For words available in GloVe vocabulary, replace the vectors
    # Others will remain random to start with and tune with the model
    embedding_matrix = torch.empty(len(word_to_index), embedding_dim)

    for word, idx in tqdm(word_to_index.items(), total=len(word_to_index), desc="preparing embedding matrix"):
        embedding_matrix[idx, :] = embedding_map.pop(word, torch.zeros(embedding_dim))

    return embedding_matrix


if __name__ == "__main__":

    with open("../data/vocab.txt", "r") as f:
        vocab = f.read().split("\n")
    word_to_index = {w: idx for (idx, w) in enumerate(vocab)}

    embedding_matrix = get_embedding_weights("../data/glove.6B.300d.txt", word_to_index, embedding_dim=300)
    print("shape of embedding matrix is ", embedding_matrix.shape)
    torch.save(embedding_matrix, "../data/glove.pt")
