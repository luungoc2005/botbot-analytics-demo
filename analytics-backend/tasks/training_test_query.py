from config import DATA_DIR, CACHE_DIR
from common import cache, utils, ignore_lists, visualization
from tasks.common import return_response
from tasks.modeling.pytorch_model import LSTMTextClassifier, CNNTextClassifier, DfTrainingDataset

from os import listdir, path, makedirs

import statistics
from scipy import stats
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

import json
import joblib

import torch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str, default='')
parser.add_argument('--query', type=str, default='')

parser.add_argument("--callback_url", type=str, default='')
parser.add_argument("--sid", type=str, default='')

args = parser.parse_args()

if __name__ == '__main__':
    print(args)

    file_name = args.file_name

    file_path = path.join(DATA_DIR, file_name)

    file_hash = cache.get_file_hash(file_path)

    if not path.exists(CACHE_DIR) or not path.isdir(CACHE_DIR):
        makedirs(CACHE_DIR)

    data_cache_path = path.join(CACHE_DIR, f'torchdata_{file_hash}.pt')

    train_dataset = torch.load(data_cache_path)

    cache_path = path.join(CACHE_DIR, f'torchmodel_{file_hash}.pt')
    # clf = LSTMTextClassifier({
    #     'embedding_size': train_dataset.embedding_size,
    #     'context_size': train_dataset.context_size,
    #     'num_classes': train_dataset.num_classes
    # }, train_dataset)

    clf = CNNTextClassifier({
        'embedding_size': train_dataset.embedding_size,
        'context_size': train_dataset.context_size,
        'num_classes': train_dataset.num_classes
    }, train_dataset)

    clf.load_state_dict(torch.load(cache_path))
    
    query = utils.tokenize_text_list([args.query])
    query_emb = torch.FloatTensor(utils.get_sentence_vectors_full(query))

    # prediction
    input_context = torch.zeros((1, train_dataset.context_size)).float()
    clf.zero_grad()
    preds_proba = torch.softmax(clf(
            query_emb, input_context
        ),
        dim=-1
    )

    preds_proba, preds_idx = torch.max(preds_proba, axis=-1)
    preds_proba = preds_proba.detach().numpy()

    attr_target = int(preds_idx.detach().item())

    from captum.attr import IntegratedGradients

    ig = IntegratedGradients(clf)
    attributions_ig, delta = ig.attribute(
        query_emb,
        target=attr_target,
        additional_forward_args=(input_context,),
        n_steps=500, 
        return_convergence_delta=True
    )

    output_attr = attributions_ig.detach().numpy()
    output_attr = visualization.normalize_attr(output_attr)[0]

    response = {
        'predicted': str(train_dataset.classes_[attr_target]),
        'confidence': float(preds_proba[0]),
        'attributions': [
            {
                'text': str(word),
                'value': float(output_attr[idx]),
                'color': visualization.get_color(output_attr[idx])
            }
            for idx, word in enumerate(query[0])
        ]
    }

    return_response(args, response)