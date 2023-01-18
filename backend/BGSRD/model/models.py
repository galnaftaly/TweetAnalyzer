import torch as th
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from .torch_gcn import GCN

class BertClassifier(th.nn.Module):
    def __init__(self, pretrained_model = 'roberta_base', nb_class = 20):
        super(BertClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)

    def forward(self, input_ids, attention_mask):
        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_logit = self.classifier(cls_feats)
        return cls_logit


class BertGCN(th.nn.Module):
    def __init__(self, pretrained_model = 'roberta-base', nb_class = 20, m = 0.7, gcn_layers = 2, n_hidden = 200, dropout = 0.5):
        super(BertGCN, self).__init__()
        self.m = m # The factor balancing BERT and GCN prediction
        self.nb_class = nb_class # The number of possible classes
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model) # Preparing the inputs for a model. 
        self.bert_model = AutoModel.from_pretrained(pretrained_model) # Pretrained BERT model
        self.feat_dim = list(self.bert_model.modules())[-2].out_features # Dimensionality of the encoder layers and the pooler layer.
        self.classifier = th.nn.Linear(self.feat_dim, nb_class) # Applies a linear transformation to the input data
        self.gcn = GCN(
            in_feats = self.feat_dim, # Input features size
            n_hidden = n_hidden, # The dimension of gcn hidden layer
            n_classes = nb_class, # Number of possible classes
            n_layers = gcn_layers - 1, # Number of convolution layer
            activation = F.elu, # Activation function decides whether a neuron should be activated or not
            dropout = dropout # The proportion of randomly selected neurons are ignored during training
        )

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim = 1)(cls_logit)
        # The input features for gcn is the output features of BERT
        gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        # Balancing BERT and GCN prediction
        pred = (gcn_pred + 1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        return pred
