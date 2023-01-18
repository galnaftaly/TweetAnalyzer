import os
import argparse
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from math import log
from utils import *
from collections import Counter
import pandas as pd
import logging

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', required = True)
    parser.add_argument('--word_embeddings_dim', '-wd', type = int, default = 300)
    parser.add_argument('--window_size', '-s', type = int, default = 20)
    args = parser.parse_args()
    return args.dataset, args.word_embeddings_dim, args.window_size

def prepare_data(datasets_dir, dataset):
    df = pd.read_csv(os.path.join(datasets_dir, dataset, '{}.csv'.format(dataset)), index_col = False)
    df.dropna(inplace = True)
    text_list = df.text.to_list()
    label_list = df.label.to_list()
    classes = list(set(label_list))
    train_size = len(df[df.type == 'train'])
    val_size = int(0.1 * train_size)
    real_train_size = train_size - val_size
    test_size = len(df[df.type == 'test'])
    print("len(text_list)")
    print(len(text_list))
    return text_list, label_list, classes, real_train_size, train_size, val_size, test_size

def build_vocab(sentences):
    word_freq = Counter()
    vocab = set()
    for doc_words in sentences:
        words = doc_words.split()
        for word in words:
            vocab.add(word)
            word_freq[word] += 1
    vocab_size = len(vocab)
    print("vocab_size")
    print(vocab_size)
    return list(vocab), vocab_size
    
def calculate_word_doc_frequency(sentences):
    word_doc_list = {}
    for i, doc in enumerate(sentences):
        words = doc.split()
        appeared = set()
        for word in words:
            if word in appeared:
                continue
            if word not in word_doc_list:
                word_doc_list[word] = []
            word_doc_list[word].append(i)
            appeared.add(word)
    word_doc_freq = {word : len(doc_list) for word, doc_list in word_doc_list.items()}
    return word_doc_freq


def create_doc_word_vectors(vocab_size, word_embeddings_dim, train_size, sentences, labels, classes):
    word_vectors = np.random.uniform(-0.01, 0.01, (vocab_size, word_embeddings_dim))
    row_allx, col_allx, data_allx  = [], [], []
    for i in range(train_size):
        doc_vec = np.zeros(word_embeddings_dim)
        doc_words = sentences[i]
        words = doc_words.split()
        doc_len = len(words)
        for j in range(word_embeddings_dim):
            row_allx.append(int(i))
            col_allx.append(j)
            data_allx.append(doc_vec[j] / doc_len)

    for i in range(vocab_size):
        for j in range(word_embeddings_dim):
            row_allx.append(int(i + train_size))
            col_allx.append(j)
            data_allx.append(word_vectors.item((i, j)))

    row_allx, col_allx, data_allx = np.array(row_allx), np.array(col_allx), np.array(data_allx)
    allx = sp.csr_matrix((data_allx, (row_allx, col_allx)), shape = (train_size + vocab_size, word_embeddings_dim))

    ally = []
    for i in range(train_size):
        label = labels[i]
        one_hot = np.zeros(len(classes))
        label_index = classes.index(label)
        one_hot[label_index] = 1
        ally.append(one_hot)

    for i in range(vocab_size):
        one_hot = np.zeros(len(classes))
        ally.append(one_hot)
    ally = np.array(ally)
    return allx, ally
    
def calculate_word_occurences(sentences, window_size):
    # word co-occurence with context windows
    windows = []
    for doc_words in sentences:
        words = doc_words.split()
        length = len(words)
        if length <= window_size:
            windows.append(words)
        else:
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                windows.append(window)
    return windows

def calculate_word_pair_count(windows, word_id_map):
    # word_pair_count dictionary with frequency of each pair of words in different windows
    word_pair_count = Counter()
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                word_i, word_j = window[i], window[j]
                word_i_id, word_j_id = word_id_map[word_i], word_id_map[word_j]
                if word_i_id == word_j_id:
                    continue
                # order 1 - (word_i_id, word_j_id)
                word_pair_str = str(word_i_id) + ',' + str(word_j_id)
                word_pair_count[word_pair_str] += 1
                # order 2 - (word_j_id, word_i_id)
                word_pair_str = str(word_j_id) + ',' + str(word_i_id)
                word_pair_count[word_pair_str] += 1           
    return word_pair_count

def calculate_word_window_frequency(windows):
    # word_window_freq dictionary with frequency of each word in different windows
    word_window_freq = Counter()
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            word_window_freq[window[i]] += 1
            appeared.add(window[i])
    return word_window_freq

def calculate_doc_word_frequency(sentences, word_id_map):
    doc_word_freq = Counter()
    for doc_id in range(len(sentences)):
        doc_words = sentences[doc_id]
        words = doc_words.split()
        for word in words:
            word_id = word_id_map[word]
            doc_word_str = str(doc_id) + ',' + str(word_id)
            doc_word_freq[doc_word_str] += 1
    return doc_word_freq



def main():
    dataset, word_embeddings_dim, window_size = parse_arguments()
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    log_file = os.path.join(data_dir, dataset, 'build_graph.log')
    logger = logging.getLogger(__name__)
    sh, fh = create_logger(logger, log_file)
    logger.addHandler(sh)
    logger.addHandler(fh)
    
    logger.info("Preparing data...")
    sentences, labels, classes, real_train_size, train_size, val_size, test_size = prepare_data(datasets_dir, dataset)
    logger.debug("Number of dataset records: {}".format(len(sentences)))
    
    logger.info("Building vocabulary...")
    vocab, vocab_size = build_vocab(sentences)
    logger.debug("Vocabulary size: {}".format(str(vocab_size)))
    
    logger.info("Calculating word-document frequency...")
    word_doc_freq = calculate_word_doc_frequency(sentences)
    word_id_map = {vocab[i] : i for i in range(vocab_size)}
    
    # different training rates
    row_x, col_x, data_x = [], [], []
    for i in range(real_train_size):
        doc_vec = np.zeros(word_embeddings_dim)
        doc_words = sentences[i]
        words = doc_words.split()
        for j in range(word_embeddings_dim):
            row_x.append(i)
            col_x.append(j)
            data_x.append(doc_vec[j] / len(words))
    x = sp.csr_matrix((data_x, (row_x, col_x)), shape = (real_train_size, word_embeddings_dim))
    y = []
    for i in range(real_train_size):
        label = labels[i]
        one_hot = np.zeros(len(classes))
        label_index = classes.index(label)
        one_hot[label_index] = 1
        y.append(one_hot)
    y = np.array(y)
    # tx: feature vectors of test docs, no initial features
    row_tx, col_tx, data_tx = [], [], []
    for i in range(test_size):
        doc_vec = np.zeros(word_embeddings_dim)
        doc_words = sentences[i + train_size]
        words = doc_words.split()
        for j in range(word_embeddings_dim):
            row_tx.append(i)
            col_tx.append(j)
            data_tx.append(doc_vec[j] / len(words))
    tx = sp.csr_matrix((data_tx, (row_tx, col_tx)), shape = (test_size, word_embeddings_dim))
    ty = []
    for i in range(test_size):
        label = labels[i + train_size]
        one_hot = np.zeros(len(classes))
        label_index = classes.index(label)
        one_hot[label_index] = 1
        ty.append(one_hot)
    ty = np.array(ty)
    word_vectors = np.random.uniform(-0.01, 0.01, (vocab_size, word_embeddings_dim))
    row_allx, col_allx, data_allx  = [], [], []
    for i in range(train_size):
        doc_vec = np.zeros(word_embeddings_dim)
        doc_words = sentences[i]
        words = doc_words.split()
        doc_len = len(words)
        for j in range(word_embeddings_dim):
            row_allx.append(int(i))
            col_allx.append(j)
            data_allx.append(doc_vec[j] / doc_len)
    for i in range(vocab_size):
        for j in range(word_embeddings_dim):
            row_allx.append(int(i + train_size))
            col_allx.append(j)
            data_allx.append(word_vectors.item((i, j)))
    row_allx, col_allx, data_allx = np.array(row_allx), np.array(col_allx), np.array(data_allx)
    allx = sp.csr_matrix((data_allx, (row_allx, col_allx)), shape = (train_size + vocab_size, word_embeddings_dim))
    ally = []
    for i in range(train_size):
        label = labels[i]
        one_hot = np.zeros(len(classes))
        label_index = classes.index(label)
        one_hot[label_index] = 1
        ally.append(one_hot)
    for i in range(vocab_size):
        one_hot = np.zeros(len(classes))
        ally.append(one_hot)
    ally = np.array(ally)
    
 
    logger.info("Building document-word heterogeneous graph...")
    windows = calculate_word_occurences(sentences, window_size)
    word_pair_count = calculate_word_pair_count(windows, word_id_map)
    word_window_freq = calculate_word_window_frequency(windows)
    
    logger.info("Calculating PMI...")
    row, col, weight = [], [], []
    # pmi as weights between two words
    num_window = len(windows)
    for word_pair, count in word_pair_count.items():
        pair = word_pair.split(',')
        i, j = int(pair[0]), int(pair[1])
        word_freq_i, word_freq_j = word_window_freq[vocab[i]], word_window_freq[vocab[j]]
        pij = float(count / num_window)
        pi, pj = float(word_freq_i / num_window), float(word_freq_j / num_window)
        pmi = log(pij / (pi * pj))
        # edge between two words exists if there is positive correlaction between them
        if pmi > 0:
            row.append(train_size + i)
            col.append(train_size + j)
            weight.append(pmi)
            
            
    doc_word_freq = calculate_doc_word_frequency(sentences, word_id_map)
    logger.info("Calculating TF-IDF...")
    for i, doc in enumerate(sentences):
        words = doc.split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = str(i) + ',' + str(j)
            tf = doc_word_freq[key]
            if i < train_size:
                row.append(i)
            else:
                row.append(i + vocab_size)
            col.append(train_size + j)
            idf = log(float(len(sentences)) / word_doc_freq[vocab[j]])
            weight.append(tf * idf)
            doc_word_set.add(word)

    logger.info("Saving data...")
    # number of nodes in the graph is number of unique words + number of tweeets
    node_size = train_size + vocab_size + test_size
    adj = sp.csr_matrix((weight, (row, col)), shape = (node_size, node_size))

    objects_to_dump = {'x': x, 'y': y, 'tx': tx, 'ty': ty, 'allx': allx, 'ally': ally, 'adj': adj}
    for obj_str, obj in objects_to_dump.items():
        with open(os.path.join(data_dir, dataset, 'ind.{}.{}'.format(dataset, obj_str)), 'wb') as f:
            pkl.dump(obj, f)
            
if __name__ == "__main__":
    main()