from config import DATA_DIR, CACHE_DIR
from common import cache, utils, ignore_lists, dialogflow
from tasks.common import return_response

from os import listdir, path, makedirs

import statistics
from scipy import stats
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

import json
import joblib

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
    
    cache_path = path.join(CACHE_DIR, f'model_{file_hash}.bin')

    # return cached result
    # if path.exists(cache_path) and path.isfile(cache_path):
    #     with open(cache_path, 'r') as cached_file:
    #         response = json.load(cached_file)
    #     return_response(args, response)
    #     exit()

    if file_path.lower()[-5:] == '.json':
        with open(file_path, 'r') as input_file:
            training_file = json.load(input_file)
    else:
        training_file = dialogflow.load_dialogflow_archive(file_path)

    #TODO