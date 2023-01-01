import torch as th
import torch.nn.functional as F
from utils.utils import *
import dgl
import torch.utils.data as Data
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss,Precision,Recall
import numpy as np
import os
from sklearn.metrics import accuracy_score
import argparse, shutil, logging
from torch.optim import lr_scheduler
from models.bert import BertClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type = int, default = 128, help = 'the input length for bert')
parser.add_argument('--batch_size', type = int, default = 32)
parser.add_argument('--nb_epochs', type = int, default = 60)
parser.add_argument('--bert_lr', type = float, default = 1e-3)
parser.add_argument('--dataset', default = 'cresci', choices = ['cresci', 'botometer', 'botwiki', 'cresci-stock', 'midterm'])
parser.add_argument('--bert_init', type = str, default ='roberta-base',
                    choices = ['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'])
parser.add_argument('--checkpoint_dir', default = None, help = 'checkpoint directory, [bert_init]_[dataset] if not specified')

parser = argparse.ArgumentParser()

args = parser.parse_args()

max_length = 128
batch_size = 128
nb_epochs = 60
bert_lr = 1e-3
dataset = 'ectf'
bert_init = 'roberta-base'
checkpoint_dir = None
# max_length = args.max_length
# batch_size = args.batch_size
# nb_epochs = args.nb_epochs
# bert_lr = args.bert_lr
# dataset = args.dataset
# bert_init = args.bert_init

cpu = th.device('cpu')
data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
log_file = os.path.join(data_dir, 'logs', 'finetune_bert.log')


logger = logging.getLogger(__name__)
sh, fh = create_logger(logger, log_file)
logger.addHandler(sh)
logger.addHandler(fh)

ckpt_dir = './checkpoint/{}_{}'.format(bert_init, dataset) if checkpoint_dir is None else checkpoint_dir
os.makedirs(ckpt_dir, exist_ok = True)
shutil.copy(os.path.basename(__file__), ckpt_dir)
logger.info('checkpoints will be saved in {}'.format(ckpt_dir))


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
model = BertClassifier(pretrained_model = bert_init, nb_class = nb_class)

idx_train_real = df_text[(df_text.label == 1) & (df_text.type == 'train')].index.to_list()
idx_val_real = df_text[(df_text.label == 1) & (df_text.type == 'val')].index.to_list()
test_real_index = df_text[(df_text.label == 1) & (df_text.type == 'test')].index.to_list()
idx_test_real = list(map(lambda x: x + nb_word, test_real_index))

# 1d array with 1 if the index belong to train and label == 1
y_train, y_val, y_test = np.zeros(nb_node), np.zeros(nb_node), np.zeros(nb_node)
y_train[idx_train_real] = 1
y_val[idx_val_real] = 1
y_test[idx_test_real] = 1
y = th.LongTensor(y_train + y_val + y_test)


label = {}
label['train'], label['val'], label['test'] = y[:nb_train], y[nb_train:nb_train + nb_val], y[-nb_test:]
input_ids, attention_mask = {}, {}
input_ids_, attention_mask_ = encode_input(df_text.text.to_list(), model.tokenizer, max_length, True)

# create train/test/val datasets and dataloaders
input_ids['train'], input_ids['val'], input_ids['test'] =  input_ids_[:nb_train], input_ids_[nb_train:nb_train+nb_val], input_ids_[-nb_test:]
attention_mask['train'], attention_mask['val'], attention_mask['test'] =  attention_mask_[:nb_train], attention_mask_[nb_train:nb_train+nb_val], attention_mask_[-nb_test:]

datasets = {}
loader = {}
for split in ['train', 'val', 'test']:
    datasets[split] =  Data.TensorDataset(input_ids[split], attention_mask[split], label[split])
    loader[split] = Data.DataLoader(datasets[split], batch_size = batch_size, shuffle = True)


################################################################################
################################### TRAINING ###################################
################################################################################

logger.info("Starting training process...")

# Training
optimizer = th.optim.Adam(model.parameters(), lr = bert_lr)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = [30], gamma = 0.1)

def train_step(engine, batch):
    global model, optimizer
    model.train()
    model = model.to(cpu)
    #model = model.to(gpu)
    optimizer.zero_grad()
    (input_ids, attention_mask, label) = [x.to(cpu) for x in batch]
    #(input_ids, attention_mask, label) = [x.to(gpu) for x in batch]
    optimizer.zero_grad()
    y_pred = model(input_ids, attention_mask)
    y_true = label.type(th.long)
    loss = F.cross_entropy(y_pred, y_true)
    loss.backward()
    optimizer.step()
    train_loss = loss.item()
    with th.no_grad():
        y_true = y_true.detach().cpu()
        y_pred = y_pred.argmax(axis = 1).detach().cpu()
        train_acc = accuracy_score(y_true, y_pred)
    return train_loss, train_acc


trainer = Engine(train_step)

# Define the metric
precision = Precision()
recall = Recall()

def test_step(engine, batch):
    precision.reset()
    global model
    with th.no_grad():
        model.eval()
        model = model.to(cpu)
        (input_ids, attention_mask, label) = [x.to(cpu) for x in batch]
        #(input_ids, attention_mask, label) = [x.to(gpu) for x in batch]
        optimizer.zero_grad()
        y_pred = model(input_ids, attention_mask)
        y_true = label
        precision.update((y_pred, y_true))
        recall.update((y_pred, y_true))
        return y_pred, y_true

evaluator = Engine(test_step)

metrics={
    'acc': Accuracy(),
    'nll': Loss(th.nn.CrossEntropyLoss()),
    'precision': Precision(average=False),
    'recall':  Recall(average=False)
}

for n, f in metrics.items():
    f.attach(evaluator, n)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(loader['train'])
    metrics = evaluator.state.metrics
    train_acc, train_nll = metrics["acc"], metrics["nll"]
    evaluator.run(loader['val'])
    metrics = evaluator.state.metrics
    val_acc, val_nll = metrics["acc"], metrics["nll"]
    evaluator.run(loader['test'])
    metrics = evaluator.state.metrics
    test_acc, test_nll = metrics["acc"], metrics["nll"]
    test_pre = metrics["precision"]
    test_pre = th.mean(test_pre)
    test_recall = metrics["recall"]
    test_recall = th.mean(test_recall)
    test_f1 = test_pre * test_recall * 2 / (test_pre + test_recall)

    logger.info(
        "\rEpoch: {}  Train acc: {:.4f} loss: {:.4f}  Val acc: {:.4f} loss: {:.4f}  Test acc: {:.4f} Test pre: {:.4f} Test recall: {:.4f} Test f1: {:.4f} loss: {:.4f}"
        .format(trainer.state.epoch, train_acc, train_nll, val_acc, val_nll, test_acc, test_pre, test_recall, test_f1, test_nll)
    )
    if val_acc > log_training_results.best_val_acc:
        logger.info("New checkpoint")
        th.save(
            {
                'bert_model': model.bert_model.state_dict(),
                'classifier': model.classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'
            )
        )
        log_training_results.best_val_acc = val_acc

        
log_training_results.best_val_acc = 0
trainer.run(loader['train'], max_epochs = nb_epochs)
