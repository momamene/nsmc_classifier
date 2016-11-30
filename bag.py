import tensorflow as tf
import numpy as np
import scipy.io
from konlpy.tag import Twitter
import nltk
import time

pos_tagger = Twitter()

TRAIN_PATH = './data/ratings_train.txt'
TEST_PATH = './data/ratings_test.txt'

DIST_PATH = './dist/ratings.embed.mat'

FEATURE_NUM = 2000

def read_data(path):
    with open(path, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data

def tokenize(doc):
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

def term_exists(doc, words):
    return [
        int(word in set(doc))
        for word
        in words
    ]

if __name__ == '__main__':
    print('read data') ; tic = time.time()
    train_data = read_data(TRAIN_PATH)
    test_data = read_data(TEST_PATH)

    train_data = train_data[:100]
    test_data = test_data[:100]
    print(time.time() - tic)

    print('tokenize') ; tic = time.time()
    train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
    test_docs = [(tokenize(row[1]), row[2]) for row in test_data]

    print(time.time() - tic)

    # Data exploration
    print('Data exploration') ; tic = time.time()

    tokens = [t for d in train_docs for t in d[0]]
    text = nltk.Text(tokens, name='NMSC')
    print(time.time() - tic)

    # Feature Extraction
    print('Feature Extraction with term-existance') ; tic = time.time()

    selected_words = [
        f[0]
        for f
        in text.vocab().most_common(FEATURE_NUM)
    ]

    train_xy = np.array([
        term_exists(doc, selected_words) + [label]
        for doc, label
        in train_docs
    ], dtype='int')

    test_xy = np.array([
        term_exists(doc, selected_words) + [label]
        for doc, label
        in test_docs
    ], dtype='int')

    print(time.time() - tic)

    scipy.io.savemat(
        DIST_PATH, {
            'train': train_xy,
            'test': test_xy,
        }
    )
