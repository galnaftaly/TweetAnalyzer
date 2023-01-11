import torch as th
import os
import pickle as pkl
from utils import *
import dgl
from model import BertGCN
from math import log
import torch.utils.data as Data
from ignite.engine import Events, Engine
import torch.nn.functional as F


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
with open(os.path.join(data_dir, 'twitter', 'ind.twitter.adj'), 'rb') as f:
    adj = pkl.load(f, encoding = 'latin1')

text = [
    '@TheEllenShow Please check into Salt River horses help stop the annihilation about to happen without 54000 more signatures.change .org Thx',
    'Why is there an ambulance right outside my work',  
    '@local_arsonist LMFAO',
    'http://t.co/FueRk0gWui Twelve feared killed in Pakistani air ambulance helicopter crash http://t.co/Mv7GgGlmVc',
    "Deadpool is already one of my favourite marvel characters and all I know is he wears a red suit so the bad guys can't tell if he's bleeding",
    'This night just blew up rq',
    'Zayn just blew up twitter.',
    '@UtahCanary sigh daily battle.',
    'The Threat | Anthrax | CDC http://t.co/q6oxzq45VE via @CDCgov', #1
    'To fight bioterrorism sir.',
    "Destruction magic's fine just don't go burning down any buildings.",
    "Twelve feared killed in Pakistani air ambulance helicopter crash - Reuters http://t.co/mDnUGVuBwN #yugvani"  
]


adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)
nb_node = adj.shape[0]
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
nb_word = nb_node - nb_train - nb_val - nb_test
batch_size = 64
test_idx = Data.TensorDataset(th.arange(nb_node - nb_test, nb_node, dtype = th.long))
idx_loader_test = Data.DataLoader(test_idx, batch_size = batch_size)

#test_idx = Data.TensorDataset(th.tensor(text_idx, dtype = th.long))




text_idx = [802, 942, 943, 944, 945, 946, 947, 948, 949, 971, 983, 989, 991]
nb_word = 5926
text_idx = list(map((lambda x: x + nb_word), text_idx))

cpu = th.device('cpu')
pretrained_model = 'roberta-base'
max_length = 128
nb_node = adj.shape[0]
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()



model = BertGCN(nb_class = 2, pretrained_model = pretrained_model, m = 0.7, gcn_layers = 2, n_hidden = 200, dropout = 0.5)
checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoint', 'roberta-base_gcn_twitter', 'checkpoint.pth')
checkpoint = th.load(checkpoint_path)
model.load_state_dict(checkpoint['model'])
input = checkpoint['tokenizer'](text, max_length = max_length, truncation = True, padding = 'max_length', return_tensors = 'pt')
nb_test = len(text)

with open(os.path.join(data_dir, 'twitter', 'graph_twitter'), 'rb') as f:
    g = pkl.load(f, encoding = 'latin1')

#test_idx = Data.TensorDataset(th.tensor(text_idx, dtype = th.long))
#idx_loader_test = Data.DataLoader(test_idx, batch_size = batch_size)
for i, batch in enumerate(idx_loader_test):
    with th.no_grad():
        print(batch)
        
exit(1)

for i, batch in enumerate(idx_loader_test):
    with th.no_grad():
        model.eval()
        model = model.to(cpu)
        output = model(g, batch)
        g = g.to(cpu)
        (idx, ) = [x.to(cpu) for x in batch]
        y_pred = model(g, idx)
        print(y_pred)
        y_pred = y_pred.argmax(axis=1).detach().cpu()
        print(y_pred)
exit(1)