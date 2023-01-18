import torch as th
import torch.nn.functional as F
import torch.utils.data as Data
import os
import argparse, shutil, logging
import logging
import pandas as pd
from utils import *
from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss, Precision, Recall
from sklearn.metrics import accuracy_score
from torch.optim import lr_scheduler
from model import BertClassifier


parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type = int, default = 128, help = 'the input length for bert')
parser.add_argument('--batch_size', type = int, default = 64)
parser.add_argument('--nb_epochs', type = int, default = 10)
parser.add_argument('--bert_lr', type = float, default = 1e-3)
parser.add_argument('--dataset', '-d', required = True)
parser.add_argument('--bert_init', type = str, default = 'roberta-base', choices = ['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased'])
parser.add_argument('--checkpoint_dir', default = None, help = 'checkpoint directory, [bert_init]_[dataset] if not specified')

args = parser.parse_args()

max_length = args.max_length # The maximum length (in number of tokens) for the inputs to the transformer model
batch_size = args.batch_size # The number of training examples utilized in one iteration
nb_epochs = args.nb_epochs # The number of complete passes through the training dataset
bert_lr = args.bert_lr # Tuning parameter that determines the step size at each iteration while moving toward a minimum of a loss function
dataset = args.dataset
bert_init = args.bert_init # Pretrained BERT model
checkpoint_dir = args.checkpoint_dir
datasets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
pretrained_bert_ckpt_file = str('./checkpoint/{}_{}/checkpoint.pth'.format(bert_init, dataset))
pretrained_bert_ckpt = pretrained_bert_ckpt_file if os.path.isfile(pretrained_bert_ckpt_file) else None

ckpt_dir = './checkpoint/{}_{}'.format(bert_init, dataset) if checkpoint_dir is None else checkpoint_dir
os.makedirs(ckpt_dir, exist_ok = True)
shutil.copy(os.path.basename(__file__), ckpt_dir)
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
log_file = os.path.join(ckpt_dir, 'finetune_bert.log')
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
y_train, y_val, y_test: n*c matrices 
train_mask, val_mask, test_mask: n-d bool array
train_size, test_size: unused
'''

# Compute number of real train/val/test/word nodes and number of classes
nb_node = adj.shape[0]
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
nb_word = nb_node - nb_train - nb_val - nb_test
nb_class = y_train.shape[1]

# Instantiate model according to number of classes
model = BertClassifier(pretrained_model = bert_init, nb_class = nb_class)

# Transform one-hot label to class ID for pytorch computation
y = th.LongTensor((y_train + y_val + y_test).argmax(axis = 1))
label = {}
label['train'], label['val'], label['test'] = y[:nb_train], y[nb_train:nb_train + nb_val], y[-nb_test:]

# Load documents and compute input encodings
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
df = pd.read_csv(os.path.join(datasets_dir, dataset, '{}.csv'.format(dataset)), index_col = False)
df.dropna(inplace = True)
text_list = df.text.to_list()

# Representation of the text as a numeric vectors - word embedding
def encode_input(text, tokenizer):
    input = tokenizer(text, max_length = max_length, truncation = True, padding = True, return_tensors = 'pt')
    return input.input_ids, input.attention_mask

input_ids, attention_mask = {}, {}
input_ids_, attention_mask_ = encode_input(text_list, model.tokenizer)

# Create train/test/val datasets and dataloaders
input_ids['train'], input_ids['val'], input_ids['test'] =  input_ids_[:nb_train], input_ids_[nb_train:nb_train+nb_val], input_ids_[-nb_test:]
attention_mask['train'], attention_mask['val'], attention_mask['test'] =  attention_mask_[:nb_train], attention_mask_[nb_train:nb_train+nb_val], attention_mask_[-nb_test:]

datasets = {}
loader = {}
for split in ['train', 'val', 'test']:
    datasets[split] =  Data.TensorDataset(input_ids[split], attention_mask[split], label[split])
    loader[split] = Data.DataLoader(datasets[split], batch_size=batch_size, shuffle=True)


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

optimizer = th.optim.Adam(model.parameters(), lr = bert_lr)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma = 0.1)

def train_step(engine, batch):
    global model, optimizer
    model.train()
    #model = model.to(cpu)
    model = model.to(gpu)
    optimizer.zero_grad()
    #(input_ids, attention_mask, label) = [x.to(cpu) for x in batch]
    (input_ids, attention_mask, label) = [x.to(gpu) for x in batch]
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
        #model = model.to(cpu)
        model = model.to(gpu)
        #(input_ids, attention_mask, label) = [x.to(cpu) for x in batch]
        (input_ids, attention_mask, label) = [x.to(gpu) for x in batch]
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
                'model': model.state_dict(),
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
