from __future__ import print_function
import gensim.downloader as api # package to download text corpus
import nltk # text processing
from nltk.corpus import stopwords
import string
import pickle

from collections import Counter
import random
from tqdm import tqdm
import numpy as np

# download stopwords
nltk.download('stopwords')

# download textcorpus
data = api.load('text8')

# collect all words to be removed
stop = stopwords.words('english') + list(string.punctuation)

actual_words = []
cleaned_words = []
unique_words = set()

# remove stop words
print('removing stop words from text corpus')
for words in data:
    current_nonstop_words = [w for w in words if w not in stop]
    cleaned_words += current_nonstop_words
    actual_words += words

    for ns in current_nonstop_words:
        unique_words.add(ns)

# print statistics
print(len(actual_words), 'words BEFORE cleaning stop words and punctuations')
print(len(cleaned_words), 'words AFTER cleaning stop words and punctuations')

# 'cleaned_words' and 'unique_words' to create a word2vec model

# BELOW STEPS ARE IMPLEMENTED BY ME
# This will help us reduce vocab_size to fit into gpu-memory

def subsample_frequent_words(corpus):
    filtered_corpus = []
    word_counts = dict(Counter(corpus))
    count = word_counts
    sum_word_counts = sum(list(word_counts.values()))
    word_counts = {word: word_counts[word]/float(sum_word_counts) for word in word_counts}
    for word in tqdm(corpus, total=len(corpus)):
        if random.random() < (1+np.sqrt(word_counts[word] * 1e3)) * 1e-3 / float(word_counts[word]):
            filtered_corpus.append(word)
    return filtered_corpus, count


corpus, word_counts = subsample_frequent_words(cleaned_words)
# lets remove the rare words
corpus = [c for c in corpus if word_counts[c]>8]

vocab = set()
for token in corpus:
  vocab.update([token])
print(len(corpus), len(vocab))

with open("data/vocab.txt", "w") as file:
    file.write("\n".join(vocab))

with open("data/text8.txt", "w") as file:
    file.write(" ".join(corpus))
