import numpy as np
import pickle as pkl
import scipy.sparse as sp
import os
import pandas as pd
import logging

def create_logger(logger, log_file = None):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok = True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
    return sh, fh


def load_pickle(filename):
    with open(filename, 'rb') as pkl_file:
        data = pkl.load(pkl_file)
    return data


def save_as_pickle(filename, data):
    with open(filename, 'wb') as output:
        pkl.dump(data, output)


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_corpus(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj']
    objects = []
    for i in range(len(names)):
        with open("data/{}/ind.{}.{}".format(dataset_str, dataset_str, names[i]), 'rb') as f:
            objects.append(pkl.load(f, encoding = 'latin1'))

    x, y, tx, ty, allx, ally, adj = tuple(objects)

    features = sp.vstack((allx, tx)).tolil()
    labels = np.vstack((ally, ty))

    datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    df = pd.read_csv(os.path.join(datasets_dir, dataset_str, '{}.csv'.format(dataset_str)), index_col = False)
    df.dropna(inplace = True)
    
    train_size = len(df[df.type == 'train'])
    val_size = train_size - x.shape[0]
    test_size = tx.shape[0]

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    idx_test = range(allx.shape[0], allx.shape[0] + test_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
