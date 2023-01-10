# TweetAnalyzer

## Introduction

In this project, we propose BGSRD model, which proposed a symmetric combination between BERT and Text GCN. BGSRD constuct a heterogeneous graph over the dataset
and represents documents as nodes using BERT representations. By jointly training the BERT and GCN modules within BGSRD, the proposed model is able to leverage the advantages of both worlds: large-scale pretraining which takes the advantage of the massive amount of raw data and transductive learning which jointly learns representations.

## Main Results

## Dependencies

Create virtual environment and install required packages for BGSRD model using pip:

`pip install -r requirements.txt`

## Usage

1. Run `python build_graph.py [dataset]` to build the text graph.

2. Run `python finetune_bert.py --dataset [dataset]` 
to finetune the BERT model over target dataset. The model and training logs will be saved to `checkpoint/[dataset]/[bert_init]_[dataset]/` by default. 
Run `python finetune_bert.py -h` to see the full list of hyperparameters.

3. Run `python train_bert_gcn.py --dataset [dataset] --pretrained_bert_ckpt [pretrained_bert_ckpt] -m [m]`
to train the BertGCN.
`[m]` is the factor balancing BERT and GCN prediction \(lambda in the paper\). 
The model and training logs will be saved to `checkpoint/[dataset]/[bert_init]_[gcn_model]_[dataset]/` by default. 
Run `python train_bert_gcn.py -h` to see the full list of hyperparameters.

## Acknowledgement
