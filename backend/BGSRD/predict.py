import torch as th
import os
import pickle as pkl
from utils import *
import dgl
import numpy as np
from model import BertGCN
from math import log
import torch.utils.data as Data
from ignite.engine import Events, Engine
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import io


########################################
###  #     #  #######  #######  ######   
 #   ##    #  #        #        #     #  
 #   # #   #  #        #        #     #  
 #   #  #  #  #####    #####    ######   
 #   #   # #  #        #        #   #    
 #   #    ##  #        #        #    #   
###  #     #  #        #######  #     # 
########################################

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoint')
#dataset = 'MR'
dataset = 'shakespeare'
batch_size = 64
cpu = th.device('cpu')
pretrained_model = 'roberta-base'
max_length = 128

class CPU_Unpickler(pkl.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: th.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)



def data_preprocess(documents, nb_word, df, dataset):
    #with open(os.path.join(checkpoint_dir, 'roberta-base_gcn_{}'.format(dataset), 'idx_pred_dict.pkl'), 'rb') as f:
    #    idx_pred_dict = pkl.load(f, encoding = 'latin1')
    #for idx in real_document_idx:
    #    print("pred") 
    #    print(idx_pred_dict[idx])

    def calc_book_prediction(documents):
        idx = []
        result_dict = dict.fromkeys(documents, 0)
        for book in documents:
            idx_start = int(df[df['title'] == book].index[0])
            idx_end = int(df[df['title'] == book].index[-1])
            idx.extend([i for i in range(idx_start, idx_end + 1)])
        return idx
        #    book_size = len(df[df['title'] == book])
        #    fake_prop = count_fake / book_size
        #    result_dict[book] = {'label': 1 if fake_prop > 0.5 else 0, 'accuracy': fake_prop if fake_prop > 0.5 else 1 - fake_prop}
        #return result_dict

    real_document_idx = []
    if dataset == 'shakespeare':
        real_document_idx = calc_book_prediction(documents)
    else:
        for doc in documents:
            real_document_idx.extend(df.index[df.text.str.contains(doc)].tolist())

    document_idx = th.tensor(list(map((lambda x: x + nb_word), real_document_idx)), dtype = th.long)
    test_idx = Data.TensorDataset(document_idx)
    idx_loader_test = Data.DataLoader(test_idx, batch_size = batch_size)
    return idx_loader_test


def get_prediction(documents, dataset):
    dataset_path = os.path.join(datasets_dir, dataset, '{}.csv'.format(dataset))
    df = pd.read_csv(dataset_path, index_col = False)
    model = BertGCN(nb_class = 2, pretrained_model = pretrained_model, m = 0.8, gcn_layers = 2, n_hidden = 200, dropout = 0.5)
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoint', 'roberta-base_gcn_{}'.format(dataset), 'checkpoint.pth')
    checkpoint = th.load(checkpoint_path, map_location = cpu)
    model.load_state_dict(checkpoint['model'])
    graph_path = os.path.join(os.path.dirname(__file__), 'checkpoint', 'roberta-base_gcn_{}'.format(dataset), 'graph_{}'.format(dataset))
    g = CPU_Unpickler(open(graph_path, "rb")).load()
    nb_word = 79923 ######### TODO
    idx_loader_test = data_preprocess(documents, nb_word, df, dataset)
    all_y_pred = []
    all_accuracy_list = []
    for i, batch in enumerate(idx_loader_test):
        with th.no_grad():
            model.eval()
            model = model.to(cpu)
            g = g.to(cpu)
            (idx, ) = [x.to(cpu) for x in batch]
            y_pred = model(g, idx)
            y_pred_normalized = th.nn.Softmax(dim = 1)(y_pred)
            y_pred = y_pred.argmax(axis = 1).detach().cpu()
            (y_pred_normalized, _) = th.max(y_pred_normalized, 1)
            y_accuracy_list = y_pred_normalized.tolist()
            y_pred = y_pred.tolist()
            all_y_pred.extend(y_pred)
            all_accuracy_list.extend(y_accuracy_list)
    if dataset != 'shakespeare':
        return all_y_pred, all_accuracy_list

    result_dict = {}
    start_idx = 0
    for book in documents:
        book_size = len(df[df['title'] == book])
        count_fake = np.count_nonzero(all_y_pred[start_idx:start_idx + book_size])
        fake_prop = count_fake / book_size
        result_dict[book] = {'label': 1 if fake_prop > 0.5 else 0, 'accuracy': fake_prop if fake_prop > 0.5 else 1 - fake_prop}
        start_idx += book_size
    all_books_pred = []
    all_book_accuracy = []
    for k, v in result_dict.items():
        all_books_pred.append(v['label'])
        all_book_accuracy.append(round(v['accuracy'], 3))
    return all_books_pred, all_book_accuracy 
    

