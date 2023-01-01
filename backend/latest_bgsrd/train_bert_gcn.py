import os
import numpy as np
import scipy.sparse as sp
from models.bert_gcn import BertGCN
import torch as th 
from utils.graph_utils import *
import dgl
import torch.utils.data as Data
from utils.utils import *
from models.bert_gcn import BertGCN
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss, Precision, Recall
import logging
import shutil
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score
import gc

max_length = 128
batch_size = 128
m = 0.8
nb_epochs = 50
bert_init = "roberta-base"
dataset = 'ectf'
pretrained_bert_ckpt_file = str('./checkpoint/{}_{}.checkpoint.pth'.format(bert_init, dataset))
pretrained_bert_ckpt = pretrained_bert_ckpt_file if os.path.isfile(pretrained_bert_ckpt_file) else None
checkpoint_dir = None
gcn_layers = 2
n_hidden = 200
heads = 8
dropout = 0.5
gcn_lr = 1e-3
bert_lr = 1e-5
nb_class = 2

cpu = th.device('cpu')
data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
log_file = os.path.join(data_dir, 'logs', 'train_bert_gcn.log')

logger = logging.getLogger(__name__)
sh, fh = create_logger(logger, log_file)
logger.addHandler(sh)
logger.addHandler(fh)

ckpt_dir = './checkpoint/{}_{}_{}'.format(bert_init, 'gcn', dataset) if checkpoint_dir is None else checkpoint_dir
os.makedirs(ckpt_dir, exist_ok = True)
shutil.copy(os.path.basename(__file__), ckpt_dir)
logger.info('checkpoints will be saved in {}'.format(ckpt_dir))


# create mask
def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype = np.bool)

# load documents and compute input encodings
logger.info("Loading and preparing data...")
df_text = load_pickle(os.path.join(data_dir, 'df_data.pkl'))
df_text['text'] = df_text['text'].apply(lambda x: ' '.join(x))
G_dict = load_pickle(os.path.join(data_dir, "text_graph.pkl"))
G = G_dict["graph"]

nb_node = G.number_of_nodes()
nb_train, nb_val, nb_test = len(df_text[df_text.type == 'train']), len(df_text[df_text.type == 'val']), len(df_text[df_text.type == 'test'])
nb_document = nb_train + nb_val + nb_test 
nb_word = nb_node - nb_document
nb_class = df_text['label'].nunique() # number of classes
logger.debug("Number of graph nodes: {} (No. of document: {}, word nodes: {}".format(nb_node, nb_document, nb_word))
logger.debug("Number of classes: {}".format(nb_class))
logger.debug("Train size: {}, Val size: {}, Test size: ".format(nb_train, nb_val, nb_test))

# instantiate model according to class number
model = BertGCN(nb_class = nb_class, pretrained_model = bert_init, m = m, gcn_layers = gcn_layers, n_hidden = n_hidden, dropout = dropout)
if pretrained_bert_ckpt is not None:
    ckpt = th.load(pretrained_bert_ckpt, map_location = cpu)
    model.bert_model.load_state_dict(ckpt['bert_model'])
    model.classifier.load_state_dict(ckpt['classifier'])


input_ids, attention_mask = encode_input(df_text.text.to_list(), model.tokenizer, max_length, 'max_length')
input_ids = th.cat([input_ids[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), input_ids[-nb_test:]])
attention_mask = th.cat([attention_mask[:-nb_test], th.zeros((nb_word, max_length), dtype=th.long), attention_mask[-nb_test:]])

idx_train = df_text[df_text.type == 'train'].index.to_list()
idx_val = df_text[df_text.type == 'val'].index.to_list()
idx_test = range(nb_node - nb_test, nb_node)
logger.debug("Indexes of train rows: {}".format(idx_train))
logger.debug("Indexes of validation rows: {}".format(idx_val))
logger.debug("Indexes of test rows: {}".format(idx_test))

# 1d array with true if the index is document node, otherwise false
train_mask = sample_mask(idx_train, nb_node)
val_mask = sample_mask(idx_val, nb_node)
test_mask = sample_mask(idx_test, nb_node)
doc_mask  = train_mask + val_mask + test_mask

idx_train_real = df_text[(df_text.label == 1) & (df_text.type == 'train')].index.to_list()
idx_val_real = df_text[(df_text.label == 1) & (df_text.type == 'val')].index.to_list()
test_real_index = df_text[(df_text.label == 1) & (df_text.type == 'test')].index.to_list()
idx_test_real = list(map(lambda x: x + nb_word, test_real_index))

logger.debug("Indexes of real train rows: {}".format(idx_train_real))
logger.debug("Indexes of real validation rows: {}".format(idx_val_real))
logger.debug("Indexes of real test rows: {}".format(idx_test_real))

# 1d array with 1 if the index belong to train and label == 1
y_train, y_val, y_test = np.zeros(nb_node), np.zeros(nb_node), np.zeros(nb_node)
y_train[idx_train_real] = 1
y_val[idx_val_real] = 1
y_test[idx_test_real] = 1
y = y_train + y_val + y_test

# build DGL Graph
logger.info("Building DGL graph...")
adj_norm, f = normalize_adj(G)
adj_norm_sp = sp.csr_matrix(adj_norm)
g = dgl.from_scipy(adj_norm_sp.astype('float32'), eweight_name='edge_weight')

g.ndata['input_ids'], g.ndata['attention_mask'] = input_ids, attention_mask
g.ndata['label'], g.ndata['train'], g.ndata['val'], g.ndata['test'] = \
    th.LongTensor(y), th.FloatTensor(train_mask), th.FloatTensor(val_mask), th.FloatTensor(test_mask)
g.ndata['label_train'] = th.LongTensor(y_train)
g.ndata['cls_feats'] = th.zeros((nb_node, model.feat_dim))

# create index loader
logger.info("Creating index loader...")
train_idx = Data.TensorDataset(th.arange(0, nb_train, dtype = th.long))
val_idx = Data.TensorDataset(th.arange(nb_train, nb_train + nb_val, dtype = th.long))
test_idx = Data.TensorDataset(th.arange(nb_node - nb_test, nb_node, dtype = th.long))
doc_idx = Data.ConcatDataset([train_idx, val_idx, test_idx])

idx_loader_train = Data.DataLoader(train_idx, batch_size = batch_size, shuffle = True)
idx_loader_val = Data.DataLoader(val_idx, batch_size = batch_size)
idx_loader_test = Data.DataLoader(test_idx, batch_size = batch_size)
idx_loader = Data.DataLoader(doc_idx, batch_size = batch_size, shuffle = True)

################################################################################
################################### TRAINING ###################################
################################################################################

logger.info("Starting training process...")

# Training
def update_feature():
    global model, g, doc_mask
    # no gradient needed, uses a large batchsize to speed up the process
    dataloader = Data.DataLoader(
        Data.TensorDataset(g.ndata['input_ids'][doc_mask], g.ndata['attention_mask'][doc_mask]),
        batch_size=1024
    )
    with th.no_grad():
        model = model.to(cpu)
        #model = model.to(gpu)
        model.eval()
        cls_list = []
        for i, batch in enumerate(dataloader):
            #input_ids, attention_mask = [x.to(gpu) for x in batch]
            input_ids, attention_mask = [x.to(cpu) for x in batch]
            output = model.bert_model(input_ids = input_ids, attention_mask=attention_mask)[0][:, 0]
            cls_list.append(output.cpu())
        cls_feat = th.cat(cls_list, axis = 0)
    g = g.to(cpu)
    g.ndata['cls_feats'][doc_mask] = cls_feat
    return g


optimizer = th.optim.Adam([
        {'params': model.bert_model.parameters(), 'lr': bert_lr},
        {'params': model.classifier.parameters(), 'lr': bert_lr},
        {'params': model.gcn.parameters(), 'lr': gcn_lr},
    ], lr=1e-3
)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)


def train_step(engine, batch):
    global model, g, optimizer
    model.train()
    model = model.to(cpu)
    #model = model.to(gpu)
    g = g.to(cpu)
    #g = g.to(gpu)
    optimizer.zero_grad()
    #(idx, ) = [x.to(gpu) for x in batch]
    (idx, ) = [x.to(cpu) for x in batch]
    optimizer.zero_grad()
    train_mask = g.ndata['train'][idx].type(th.BoolTensor)
    y_pred = model(g, idx)[train_mask]
    y_true = g.ndata['label_train'][idx][train_mask]
    loss = F.nll_loss(y_pred, y_true)
    loss.backward()
    optimizer.step()
    g.ndata['cls_feats'].detach_()
    train_loss = loss.item()
    with th.no_grad():
        y_true = y_true.detach().cpu()
        y_pred = y_pred.argmax(axis=1).detach().cpu()
        train_acc = accuracy_score(y_true, y_pred)
    return train_loss, train_acc


trainer = Engine(train_step)


@trainer.on(Events.EPOCH_COMPLETED)
def reset_graph(trainer):
    scheduler.step()
    update_feature()
    #th.cuda.empty_cache()

# Define the metric
precision = Precision()
def test_step(engine, batch):
    global model, g
    with th.no_grad():
        model.eval()
        model = model.to(cpu)
        #model = model.to(gpu)
        g = g.to(cpu)
        #g = g.to(gpu)
        #(idx, ) = [x.to(gpu) for x in batch]
        (idx, ) = [x.to(cpu) for x in batch]
        y_pred = model(g, idx)
        y_true = g.ndata['label'][idx]
        precision.update((y_pred, y_true))
        return y_pred, y_true


evaluator = Engine(test_step)
metrics={
    'acc': Accuracy(),
    'precision': Precision(average=False),
    'recall':  Recall(average=False),
    'nll': Loss(th.nn.NLLLoss())
}
for n, f in metrics.items():
    f.attach(evaluator, n)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(idx_loader_train)
    metrics = evaluator.state.metrics
    train_acc, train_nll = metrics["acc"], metrics["nll"]
    evaluator.run(idx_loader_val)
    metrics = evaluator.state.metrics
    val_acc, val_nll = metrics["acc"], metrics["nll"]
    evaluator.run(idx_loader_test)
    metrics = evaluator.state.metrics
    test_acc, test_nll = metrics["acc"], metrics["nll"]
    test_pre = metrics["precision"]
    test_pre = th.mean(test_pre)
    test_recall = metrics["recall"]
    test_recall = th.mean(test_recall)
    test_f1 = test_pre * test_recall * 2 / (test_pre + test_recall)
    logger.info(
        "Epoch: {}  Train acc: {:.4f} loss: {:.4f}  Val acc: {:.4f} loss: {:.4f}  Test acc: {:.4f} Test pre: {:.4f} Test recall: {:.4f} Test f1: {:.4f} loss: {:.4f}"
        .format(trainer.state.epoch, train_acc, train_nll, val_acc, val_nll, test_acc, test_pre, test_recall, test_f1, test_nll)
    )
    if val_acc > log_training_results.best_val_acc:
        logger.info("New checkpoint")
        th.save(
            {
                'bert_model': model.bert_model.state_dict(),
                'classifier': model.classifier.state_dict(),
                'gcn': model.gcn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'
            )
        )
        log_training_results.best_val_acc = val_acc
    gc.collect()


log_training_results.best_val_acc = 0
g = update_feature()
trainer.run(idx_loader, max_epochs = nb_epochs)



