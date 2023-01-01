import pickle as pkl
import logging

def load_pickle(filename):
    with open(filename, 'rb') as pkl_file:
        data = pkl.load(pkl_file)
    return data

def save_as_pickle(filename, data):
    with open(filename, 'wb') as output:
        pkl.dump(data, output)

def create_logger(logger, log_file = None):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
    return sh, fh

def encode_input(text, tokenizer, max_length, padding):
    input = tokenizer(text, max_length = max_length, truncation = True, padding = padding, return_tensors = 'pt')
    return input.input_ids, input.attention_mask