from models.bert_gcn import BertGCN

max_length = 128
batch_size = 128
#batch_size = 2
m = 0.8
nb_epochs = 50
#nb_epochs = 2
bert_init = "roberta-base"
pretrained_bert_ckpt = None
checkpoint_dir = None
gcn_layers = 2
n_hidden = 200
heads = 8
dropout = 0.5
gcn_lr = 1e-3
bert_lr = 1e-5
nb_class = 2

def get_model():
    model = BertGCN(nb_class = nb_class, pretrained_model = bert_init, m = m, gcn_layers = gcn_layers, n_hidden = n_hidden, dropout = dropout)
    return model