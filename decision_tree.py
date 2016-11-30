import tensorflow as tf
import numpy as np
from konlpy.tag import Twitter
import nltk
import time
from gensim.models import Word2Vec

pos_tagger = Twitter()

TRAIN_PATH = './data/ratings_train.tokenized.txt'
TEST_PATH = './data/ratings_test.tokenized.txt'

FEATURE_NUM = 2000

def read_data(path):
    with open(path, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data

def tokenize(doc):
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]


def term_exists(doc, words):
    return {
        'exists({})'.format(word): (word in set(doc))
        for word in selected_words
    }

if __name__ == '__main__':
    print('read data') ; tic = time.time()
    train_docs = read_data(TRAIN_PATH)
    test_docs = read_data(TEST_PATH)

    print(time.time() - tic)

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

    train_docs = train_docs[:10000]

    train_xy = [
        (term_exists(doc, selected_words), label)
        for doc, label
        in train_docs
    ]

    test_xy = [
        (term_exists(doc, selected_words), label)
        for doc, label
        in test_docs
    ]

    print(time.time() - tic)

    print('DecisionTreeClassifier') ; tic = time.time()

    classifier = nltk.DecisionTreeClassifier.train(train_xy)
    print(nltk.classify.accuracy(classifier, test_xy))
    # => 0.80418
    classifier.show_most_informative_features(10)

    print(time.time() - tic)
