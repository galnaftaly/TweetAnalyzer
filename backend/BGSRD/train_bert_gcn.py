import torch as th
import torch.nn.functional as F
import dgl
import torch.utils.data as Data
import os
import shutil
import argparse
import logging
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss, Precision, Recall
from sklearn.metrics import accuracy_score
from utils import *
from torch.optim import lr_scheduler
from model import BertGCN

parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type = int, default = 128, help = 'the input length for bert')
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('-m', '--m', type = float, default = 0.8, help = 'the factor balancing BERT and GCN prediction')
parser.add_argument('--nb_epochs', type = int, default = 50)
parser.add_argument('--bert_init', type = str, default = 'roberta-base', choices = ['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'])
parser.add_argument('--pretrained_bert_ckpt', default = None)
parser.add_argument('--dataset', '-d', required = True)
parser.add_argument('--checkpoint_dir', default = None, help = 'checkpoint directory, [bert_init]_[gcn_model]_[dataset] if not specified')
parser.add_argument('--gcn_layers', type = int, default = 2)
parser.add_argument('--n_hidden', type = int, default = 200, help = 'the dimension of gcn hidden layer')
parser.add_argument('--dropout', type = float, default = 0.5)
parser.add_argument('--gcn_lr', type = float, default = 1e-3)
parser.add_argument('--bert_lr', type = float, default = 1e-5)


args = parser.parse_args()
max_length = args.max_length # The maximum length (in number of tokens) for the inputs to the transformer model
batch_size = args.batch_size # The number of training examples utilized in one iteration
m = args.m # The factor balancing BERT and GCN prediction
nb_epochs = args.nb_epochs # The number of complete passes through the training dataset
bert_init = args.bert_init
dataset = args.dataset
datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
pretrained_bert_ckpt = args.pretrained_bert_ckpt
checkpoint_dir = args.checkpoint_dir
gcn_layers = args.gcn_layers # The number of gcn layers
n_hidden = args.n_hidden # The dimension of gcn hidden layer
dropout = args.dropout # The proportion of randomly selected neurons are ignored during training
# Tuning parameter that determines the step size at each iteration while moving toward a minimum of a loss function
gcn_lr = args.gcn_lr
bert_lr = args.bert_lr

best_batches = th.tensor([])
best_test_pred = th.tensor([])
batches = th.tensor([])
test_pred = th.tensor([])


ckpt_dir = './checkpoint/{}_gcn_{}'.format(bert_init, dataset) if checkpoint_dir is None else checkpoint_dir
os.makedirs(ckpt_dir, exist_ok = True)
shutil.copy(os.path.basename(__file__), ckpt_dir)
log_file = os.path.join(ckpt_dir, 'train_bert_gcn.log')
cpu = th.device('cpu')
gpu = th.device('cuda:0')

logger = logging.getLogger(__name__)
sh, fh = create_logger(logger, log_file)
logger.addHandler(sh)
logger.addHandler(fh)

logger.info('Arguments:')
logger.info(str(args))
logger.info('Checkpoints will be saved in {}'.format(ckpt_dir))

# Data Preprocess
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)
'''
adj: n*n sparse adjacency matrix
y_train, y_val, y_test: n*c matrices
train_mask, val_mask, test_mask: n-d bool array
'''

# Compute number of real train/val/test/word nodes and number of classes
nb_node = features.shape[0]
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
nb_word = nb_node - nb_train - nb_val - nb_test
nb_class = y_train.shape[1]

# Instantiate model according to class number
model = BertGCN(nb_class = nb_class, pretrained_model = bert_init, m = m, gcn_layers = gcn_layers, n_hidden = n_hidden, dropout = dropout)

# Load the finetuning model checkpoint we saved when ran finetune_bert
if pretrained_bert_ckpt is not None:
    #ckpt = th.load(pretrained_bert_ckpt, map_location = cpu)
    ckpt = th.load(pretrained_bert_ckpt, map_location = gpu)
    model.bert_model.load_state_dict(ckpt['bert_model'])
    model.classifier.load_state_dict(ckpt['classifier'])


df = pd.read_csv(os.path.join(datasets_dir, dataset, '{}.csv'.format(dataset)), index_col = False)
df.dropna(inplace = True)
text_list = df.text.to_list()

# Representation of the text as a numeric vectors - word embedding
def encode_input(text, tokenizer):
    input = tokenizer(text, max_length = max_length, truncation = True, padding = 'max_length', return_tensors = 'pt')
    return input.input_ids, input.attention_mask

input_ids, attention_mask = encode_input(text_list, model.tokenizer)
input_ids = th.cat([input_ids[:-nb_test], th.zeros((nb_word, max_length), dtype = th.long), input_ids[-nb_test:]])
attention_mask = th.cat([attention_mask[:-nb_test], th.zeros((nb_word, max_length), dtype = th.long), attention_mask[-nb_test:]])

# Transform one-hot label to class ID for pytorch computation
y = y_train + y_test + y_val
y_train = y_train.argmax(axis = 1)
y = y.argmax(axis = 1)

# Document mask used for update feature
doc_mask = train_mask + val_mask + test_mask

# build DGL Graph
adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
g = dgl.from_scipy(adj_norm.astype('float32'), eweight_name = 'edge_weight')
g.ndata['input_ids'], g.ndata['attention_mask'] = input_ids, attention_mask
g.ndata['label'], g.ndata['train'], g.ndata['val'], g.ndata['test'] = \
    th.LongTensor(y), th.FloatTensor(train_mask), th.FloatTensor(val_mask), th.FloatTensor(test_mask)
g.ndata['label_train'] = th.LongTensor(y_train)
g.ndata['cls_feats'] = th.zeros((nb_node, model.feat_dim))

logger.info('Graph information:')
logger.info(str(g))

# Create index loader
train_idx = Data.TensorDataset(th.arange(0, nb_train, dtype = th.long))
val_idx = Data.TensorDataset(th.arange(nb_train, nb_train + nb_val, dtype = th.long))
test_idx = Data.TensorDataset(th.arange(nb_node - nb_test, nb_node, dtype = th.long))
doc_idx = Data.ConcatDataset([train_idx, val_idx, test_idx])

idx_loader_train = Data.DataLoader(train_idx, batch_size = batch_size, shuffle = True)
idx_loader_val = Data.DataLoader(val_idx, batch_size = batch_size)
idx_loader_test = Data.DataLoader(test_idx, batch_size = batch_size)
idx_loader = Data.DataLoader(doc_idx, batch_size = batch_size, shuffle = True)


################################################################
#######  ######      #     ###  #     #  ###  #     #   #####   
   #     #     #    # #     #   ##    #   #   ##    #  #     #  
   #     #     #   #   #    #   # #   #   #   # #   #  #        
   #     ######   #     #   #   #  #  #   #   #  #  #  #  ####  
   #     #   #    #######   #   #   # #   #   #   # #  #     #  
   #     #    #   #     #   #   #    ##   #   #    ##  #     #  
   #     #     #  #     #  ###  #     #  ###  #     #   #####   
################################################################

logger.info("Starting training process...")

# Initalize graph with BERT output in each iteration
def update_feature():
    global model, g, doc_mask
    dataloader = Data.DataLoader(
        Data.TensorDataset(g.ndata['input_ids'][doc_mask], g.ndata['attention_mask'][doc_mask]),
        batch_size = 1024
    )
    with th.no_grad():
        #model = model.to(cpu)
        model = model.to(gpu)
        model.eval()
        cls_list = []
        for i, batch in enumerate(dataloader):
            #input_ids, attention_mask = [x.to(cpu) for x in batch]
            input_ids, attention_mask = [x.to(gpu) for x in batch]
            output = model.bert_model(input_ids = input_ids, attention_mask = attention_mask)[0][:, 0]
            cls_list.append(output.cpu())
        cls_feat = th.cat(cls_list, axis = 0)
    g = g.to(cpu)
    g.ndata['cls_feats'][doc_mask] = cls_feat
    return g


optimizer = th.optim.Adam([
        {'params': model.bert_model.parameters(), 'lr': bert_lr},
        {'params': model.classifier.parameters(), 'lr': bert_lr},
        {'params': model.gcn.parameters(), 'lr': gcn_lr},
    ], lr = 1e-3
)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = [30], gamma = 0.1)


def train_step(engine, batch):
    global model, g, optimizer
    model.train()
    #model = model.to(cpu)
    model = model.to(gpu)
    #g = g.to(cpu)
    g = g.to(gpu)
    optimizer.zero_grad()
    #(idx, ) = [x.to(cpu) for x in batch]
    (idx, ) = [x.to(gpu) for x in batch]
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
        y_pred = y_pred.argmax(axis = 1).detach().cpu()
        train_acc = accuracy_score(y_true, y_pred)
    return train_loss, train_acc


trainer = Engine(train_step)


@trainer.on(Events.EPOCH_COMPLETED)
def reset_graph(trainer):
    scheduler.step()
    update_feature()
    th.cuda.empty_cache()

# Define the metric
precision = Precision()
def test_step(engine, batch):
    global model, g
    with th.no_grad():
        model.eval()
        #model = model.to(cpu)
        model = model.to(gpu)
        #g = g.to(cpu)
        g = g.to(gpu)
        #(idx, ) = [x.to(cpu) for x in batch]
        (idx, ) = [x.to(gpu) for x in batch]
        y_pred = model(g, idx)
        y_true = g.ndata['label'][idx]
        precision.update((y_pred, y_true))
        return y_pred, y_true

def test_step2(engine, batch):
    global model, g, batches, test_pred 
    with th.no_grad():
        model.eval()
        #model = model.to(cpu)
        model = model.to(gpu)
        g = g.to(gpu)
        #g = g.to(cpu)
        (idx, ) = [x.to(gpu) for x in batch]
        #(idx, ) = [x.to(cpu) for x in batch]
        y_pred = model(g, idx)
        y_true = g.ndata['label'][idx]
        current_pred = y_pred.argmax(axis = 1).detach().cpu()
        batches = th.cat((batches, batch[0]), 0)
        test_pred = th.cat((test_pred, current_pred), 0)
        precision.update((y_pred, y_true))
        return y_pred, y_true

evaluator = Engine(test_step)
evaluator2 = Engine(test_step2)
metrics={
    'acc': Accuracy(),
    'precision': Precision(average = False),
    'recall':  Recall(average = False),
    'nll': Loss(th.nn.NLLLoss())
}
for n, f in metrics.items():
    f.attach(evaluator, n)
    f.attach(evaluator2, n)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    global batches, test_pred, best_batches, best_test_pred
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
        best_batches = batches
        best_test_pred = test_pred
        logger.info("New checkpoint")
        th.save({
                'model': model.state_dict(),
                'bert_model': model.bert_model.state_dict(),
                'classifier': model.classifier.state_dict(),
                'gcn': model.gcn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
                'tokenizer': model.tokenizer,
                'feat_dim': model.feat_dim
            },
            os.path.join(ckpt_dir, 'checkpoint.pth')            
        )
        with open(os.path.join(ckpt_dir, 'graph_{}'.format(dataset)), 'wb') as f:
            pkl.dump(g, f)
        log_training_results.best_val_acc = val_acc

log_training_results.best_val_acc = 0
g = update_feature()
trainer.run(idx_loader, max_epochs = nb_epochs)

def calc_book_prediction(idx_pred_dict):
    test_books = list(df[df.type == 'test'].title.unique())
    result_dict = dict.fromkeys(test_books, 0)
    for book in test_books:
        idx_start = int(df[df['title'] == book].index[0])
        idx_end = int(df[df['title'] == book].index[-1])
        count_fake = sum(v for k,v in idx_pred_dict.items() if k >= idx_start and k <= idx_end)
        book_size = len(df[df['title'] == book])
        fake_prop = count_fake / book_size
        # If fake proportion is greater than 0.5 so the whole book consider fake, otherwise it consider real
        result_dict[book] = {'label': 1 if fake_prop > 0.5 else 0, 'fake props': fake_prop}
    return result_dict


if dataset == 'shakespeare':
    real_test_idx = th.tensor([])
    for batch in best_batches:
        real_test_idx = th.cat((real_test_idx, th.tensor([batch - nb_word])), 0)
    real_test_idx_list = [int(x) for x in real_test_idx.tolist()]
    test_pred_list = [int(x) for x in best_test_pred.tolist()]
    idx_pred_dict = dict(map(lambda i,j : (i,j) , real_test_idx_list, test_pred_list))
    result_dict = calc_book_prediction(idx_pred_dict)

    all_books = df[df['type'] == 'test'].title.unique().tolist()
    all_books_labels = []
    all_fake_props = []
    for book in all_books:
        all_books_labels.append(df[(df['title'] == book) & (df['type'] == 'test')].iloc[0].label)

    all_books_pred = []
    for k, v in result_dict.items():
        all_books_pred.append(v['label'])
        all_fake_props.append(round(v['fake props'], 3))
    
    book_results_df = pd.DataFrame(
    {'Book': all_books,
     'Label': all_books_labels,
     'Prediction': all_books_pred,
     'Fake props': all_fake_props
    })
    with open(os.path.join(ckpt_dir, 'books_pred.pkl'), mode = 'wb') as f:
        pkl.dump(book_results_df, f)