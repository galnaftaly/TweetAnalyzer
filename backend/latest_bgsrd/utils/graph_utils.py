from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from itertools import combinations
import math
from tqdm import tqdm
import re
import networkx as nx
import numpy as np

def filter_tokens(tokens):
    stopwords = list(set(nltk.corpus.stopwords.words("english")))
    non_alphbetical = ["’", "“", ".",",",";","&","'s", ":", "?", "!","(",")", "@","'","'m","'no","***","--","...","[","]", "#", "%", "''", "$", "+"]
    clean_tokens = []
    for token in tokens:
        token = token.lower()
        token = re.sub('[\',.;]', '', token)
        if token not in stopwords and token not in non_alphbetical and len(token.strip()) > 0:
            clean_tokens.append(token)
    return clean_tokens

def dummy_fun(doc):
    return doc

# binomial coefficient - number of options to choose subgroup size r from n from group size k
def nCr(n,r):
    f = math.factorial
    return int(f(n)/(f(r)*f(n-r)))

def word_word_edges(df_pmi):
    word_word = []
    cols = list(df_pmi.columns)
    cols = [str(word) for word in cols]
    for w1, w2 in tqdm(combinations(cols, 2), total=nCr(len(cols), 2)):
        if (df_pmi.loc[w1,w2] > 0):
            word_word.append((w1, w2, {"weight" : df_pmi.loc[w1,w2]}))
    return word_word


##### we already normalized the matrix from A --> A_hat
#def normalize_adj(adj):
#    adj = adj + sp.eye(adj.shape[0])
#    """Symmetrically normalize adjacency matrix."""
#    adj = sp.coo_matrix(adj)
#    # sum of all the rows matrix into 1d array
#    rowsum = np.array(adj.sum(1))
#    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

# Building adjacency and degree matrix
def normalize_adj(G):
    A = nx.to_numpy_matrix(G, weight = "weight")
    
    # A = degree matrix
    A = A + np.eye(G.number_of_nodes())
    degrees = [0 if d == 0 else d[1]**(-0.5) for d in G.degree(weight = None)]
    degrees = np.diag(degrees)
    X = np.eye(G.number_of_nodes()) # features are just identity matrix
    A_hat = degrees @ A @ degrees
    f = X # (n X n) X (n X n) x (n X n) X (n X n) input of net
    return A_hat, f
    #for d in G.degree(weight=None):
    #    if d == 0:
    #        degrees.append(0)
    #    else:
    #        degrees.append(d[1]**(-0.5))
    # D^-1/2
    # X = identity matrix
    # A_hat = D^-1/2 * A * D^-1/2