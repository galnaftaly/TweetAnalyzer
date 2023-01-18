# TweetAnalyzer

## Introduction

In this project, we adopt BertGCN model, which proposed a symmetric combination between BERT and Text GCN. BertGCN construct a heterogeneous graph over the dataset
and represents documents as nodes using BERT representations. By jointly training the BERT and GCN modules within BertGCN, the proposed model is able to leverage the advantages of both worlds: large-scale pretraining which takes the advantage of the massive amount of raw data and transductive learning which jointly learns representations.

## Main Results

|**Dataset** | **Accuracy** | **Precision** | **Recall** | **F1** |
| ------------ | ---- | ---- | ---- | ---- |
| *MR* | 0.8790 | 0.8820 | 0.8790 | 0.8805 |
| *Shakespeare* | 0.7696 | 0.7574 | 0.7234 | 0.7401 |
| *Twitter* | 0.8345 | 0.8469 | 0.8164 | 0.8314 |

## Dependencies

Create virtual environment and install required packages for BGSRD model using pip:

`pip install -r requirements.txt`

## Usage

### Run the model

1. Run `cd backend/BGSRD`

2. Run `python build_graph.py [dataset]` to build the text graph.

3. Run `python finetune_bert.py --dataset [dataset]` 
to finetune the BERT model over target dataset. The model and training logs will be saved to `checkpoint/[dataset]/[bert_init]_[dataset]/` by default. 
Run `python finetune_bert.py -h` to see the full list of hyperparameters.

4. Run `python train_bert_gcn.py --dataset [dataset] --pretrained_bert_ckpt [pretrained_bert_ckpt] -m [m]`
to train the BertGCN.
`[m]` is the factor balancing BERT and GCN prediction \(lambda in the paper\). 
The model and training logs will be saved to `checkpoint/[dataset]/[bert_init]_[gcn_model]_[dataset]/` by default. 
Run `python train_bert_gcn.py -h` to see the full list of hyperparameters.

### Run the GUI

1. Run: `cd backend && uvicorn api:app` to start the application server.

2. Run `npm install` to build the dependencies.

3. Run `npm start` to start the development server and open the GUI.

## Acknowledgement
