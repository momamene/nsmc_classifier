import tensorflow as tf
import numpy as np
from konlpy.tag import Twitter
import nltk
import time
from gensim.models import Word2Vec

pos_tagger = Twitter()

TRAIN_PATH = './data/ratings_train.txt'
TEST_PATH = './data/ratings_test.txt'

TRAIN_VEC_PATH = './dist/ratings_train.vec.txt'
TEST_VEC_PATH = './dist/ratings_test.vec.txt'
TRAIN_LABEL_PATH = './dist/ratings_train.label.txt'
TEST_LABEL_PATH = './dist/ratings_test.label.txt'
TRAIN_COUNT_PATH = './dist/ratings_train.count.txt'
TEST_COUNT_PATH = './dist/ratings_test.count.txt'

FEATURE_NUM = 200

def read_data(path):
    with open(path, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:]
    return data

def tokenize(doc):
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]

if __name__ == '__main__':
    print('read data') ; tic = time.time()
    train_docs = read_data(TRAIN_PATH)
    test_docs = read_data(TEST_PATH)

    print(time.time() - tic)

    print('tokenize') ; tic = time.time()
    train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
    test_docs = [(tokenize(row[1]), row[2]) for row in test_data]

    print(time.time() - tic)

    print('word embedding') ; tic = time.time()
    sentences = [
        sentence
        for sentence, _
        in train_docs
    ]

    model = Word2Vec(
        sentences, size=FEATURE_NUM,
        window=5, min_count=5, workers=8
    )

    longest_sentence_len = max([
        len([word for word in sentence if word in model])
        for sentence, _
        in train_docs + test_docs
    ])

    _train_vec = [
        [model[word] for word in sentence if word in model]
        for sentence, _
        in train_docs
    ]

    _test_vec = [
        [model[word] for word in sentence if word in model]
        for sentence, _
        in test_docs
    ]

    train_vec = np.zeros(
        (len(train_docs), longest_sentence_len, FEATURE_NUM),
        dtype='float'
    )

    train_label = np.zeros(
        (len(train_docs), longest_sentence_len),
        dtype='int'
    )

    test_vec = np.zeros(
        (len(test_docs), longest_sentence_len, FEATURE_NUM),
        dtype='float'
    )

    test_label = np.zeros(
        (len(test_docs), longest_sentence_len),
        dtype='int'
    )

    for idx, _ in enumerate(train_vec):
        if len(_train_vec[idx]) == 0:
            continue
        train_vec[idx, 0:len(_train_vec[idx])] = np.array(_train_vec[idx])
        train_label[idx] = train_docs[idx][1]

    for idx, _ in enumerate(test_vec):
        if len(_test_vec[idx]) == 0:
            continue
        test_vec[idx, 0:len(_test_vec[idx])] = np.array(_test_vec[idx])
        test_label[idx] = test_docs[idx][1]

    print(time.time() - tic)

    train_count = np.array([
        len([word for word in sentence if word in model])
        for sentence, _
        in train_docs
    ], dtype='int')

    test_count = np.array([
        len([word for word in sentence if word in model])
        for sentence, _
        in test_docs
    ], dtype='int')

    train_vec.tofile(TRAIN_VEC_PATH)
    test_vec.tofile(TEST_VEC_PATH)
    train_label.tofile(TRAIN_LABEL_PATH)
    test_label.tofile(TEST_LABEL_PATH)
    train_count.tofile(TRAIN_COUNT_PATH)
    test_count.tofile(TEST_COUNT_PATH)

    print(train_vec.shape)
    # Usage
    # np.fromfile(TRAIN_DIST_PATH)
    #   .reshape(len(train_docs), longest_sentence_len, FEATURE_NUM))
