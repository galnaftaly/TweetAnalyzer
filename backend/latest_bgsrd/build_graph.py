import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import networkx as nx
from collections import OrderedDict
from itertools import combinations
import math
from tqdm import tqdm
import logging
from utils.graph_utils import *
from utils.utils import *

# TODO we should get all paramertes as arguments
# global variables
dataset = 'ectf'
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
test_ratio = 0.2
max_vocab_len = 10000
window_size = 20
dubug = True
log_file = os.path.join(data_dir, 'logs', 'build_graph.log')

logger = logging.getLogger(__name__)
sh, fh = create_logger(logger, log_file)
logger.addHandler(sh)
logger.addHandler(fh)


def build_text_graph(df_tfidf, vocab):
    logger.info("Building graph (No. of document: {}, word nodes: {})...".format(len(df_tfidf.index), len(vocab)))
    G = nx.Graph()
    logger.info("Adding document nodes to graph...")
    logger.debug("Number of document nodes: {}".format(len(df_tfidf)))
    G.add_nodes_from(df_tfidf.index) ## document nodes
    logger.info("Adding word nodes to graph...")
    logger.debug("Number of word nodes: {}".format(len(vocab)))
    G.add_nodes_from(vocab) ## word nodes
    ### build edges between document-word pairs
    logger.info("Building document-word edges...")
    logger.debug("Number of document-word edges: {}".format(len(df_tfidf)))
    for doc in tqdm(df_tfidf.index, total = len(df_tfidf.index)):
        for word in df_tfidf.columns:
            G.add_edge(doc, word, weight = df_tfidf.loc[doc,word])
    del df_tfidf
    return G

def calculate_tfidf(df):
    logger.info("Calculating Tf-idf...")
    vectorizer = TfidfVectorizer(input="content", max_features = max_vocab_len, tokenizer = dummy_fun, preprocessor = dummy_fun)
    vectorizer.fit(df["text"])
    # initalize tfidf matrix, at this point the matrix contains only zeros
    df_tfidf = vectorizer.transform(df["text"])
    df_tfidf = df_tfidf.toarray()
    vocab = vectorizer.get_feature_names()
    vocab = np.array(vocab)
    df_tfidf = pd.DataFrame(df_tfidf, columns = vocab)
    del vectorizer
    return df_tfidf, vocab

def calculate_co_occurrences(df, vocab, window_size):
    n_i  = OrderedDict((name, 0) for name in vocab)
    word2index = OrderedDict( (name,index) for index,name in enumerate(vocab) )
    occurrences = np.zeros( (len(vocab),len(vocab)) ,dtype=np.int32)
    number_of_windows = 0 
    logger.info("Calculating co-occurences...")
    for l in tqdm(df["text"], total = len(df["text"])):
        for i in range(len(l) - window_size):
            number_of_windows += 1
            doc = set(l[i:(i + window_size)])
            for word in doc:
                n_i[word] += 1
            for w1,w2 in combinations(doc,2):
                i1 = word2index[w1]
                i2 = word2index[w2]
                occurrences[i1][i2] += 1
                occurrences[i2][i1] += 1
    del df, word2index
    return number_of_windows, occurrences, n_i


def calculate_pmi(df, vocab, window_size):
    # Find the co-occurrences:
    number_of_windows, occurrences, n_i = calculate_co_occurrences(df, vocab, window_size)
    logger.info("Calculating PMI*...")
    ### convert to PMI
    df_pij = pd.DataFrame(occurrences, index = vocab, columns = vocab) / number_of_windows
    p_i = pd.Series(n_i, index=n_i.keys()) / number_of_windows
    del occurrences, n_i, vocab
    for col in df_pij.columns:
        df_pij[col] = df_pij[col] / p_i[col]
    for row in df_pij.index:
        df_pij.loc[row,:] = df_pij.loc[row,:] / p_i[row]
    df_pij = df_pij + 1E-9
    for col in df_pij.columns:
        df_pij[col] = df_pij[col].apply(lambda x: math.log(x))
    return df_pij

  
def main():
    logger.info("Loading data...")
    df_data_path = os.path.join(data_dir, "df_data.pkl")
    text_graph_path = os.path.join(data_dir, "text_graph.pkl")
    word_word_edges_path = os.path.join(data_dir, "word_word_edges.pkl")
    if os.path.isfile(df_data_path) and os.path.isfile(text_graph_path):
        logger.info("Datasets and graph already exists...")
        return
    
    logger.info("Building datasets and graph from raw data... Note this will take quite a while...")
    logger.info("Preparing data...")
    df = load_pickle(df_data_path)
    df.dropna(inplace = True)
    # tokenize & remove funny characters
    df["text"] = df["text"].apply(lambda x: nltk.word_tokenize(x)).apply(lambda x: filter_tokens(x))
    logger.debug("Number of dataset records: {}".format(len(df)))
    # Tfidf
    df_tfidf, vocab = calculate_tfidf(df)
    # Build graph
    G = build_text_graph(df_tfidf, vocab)
    # PMI between words
    df_pmi = calculate_pmi(df, vocab, window_size)
    logger.info("Building word-word edges...")
    # Add word_word edges to the graph
    word_word = word_word_edges(df_pmi)
    G.add_edges_from(word_word)
    logger.debug("Number of word-word edges: {}".format(len(word_word)))

    logger.info("Saving data...")
    save_as_pickle(df_data_path, df)
    save_as_pickle(word_word_edges_path, word_word)
    save_as_pickle(text_graph_path, {"graph": G})
    logger.info("Done and saved!")


if __name__ == "__main__":
    main()
