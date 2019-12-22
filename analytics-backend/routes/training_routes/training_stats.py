from config import DATA_DIR
from routes.training_routes import training_routes_blueprint
from common import cache, utils, ignore_lists

from flask import jsonify, escape, request
from werkzeug.exceptions import BadRequest
from os import listdir, path

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

import json

@training_routes_blueprint.route('/training_stats')
def training_stats():
    file_name = request.args.get("file", "")
    
    file_path = path.join(DATA_DIR, escape(file_name))
    if not path.exists(file_path) or not path.isfile(file_path):
        return BadRequest('File not found')

    with open(file_path, 'r') as input_file:
        training_file = json.load(input_file)

    # extract raw intents and labels from file. Needs upgrade
    raw_examples = []
    raw_labels = []

    for intent in training_file:
        for usersay in intent['usersays']:
            raw_labels.append(intent['name'])
            raw_examples.append(usersay)

    raw_exampes_tokens = utils.tokenize_text_list(raw_examples)
    
    le = LabelEncoder()
    X_train = utils.get_sentence_vectors(raw_exampes_tokens)
    y_train = le.fit_transform(raw_labels)

    clf = MLPClassifier(
        hidden_layer_sizes=(50,), 
        random_state=1,
        batch_size=min(32, len(X_train))
        max_iter=100)

    clf.fit(X_train, y_train)

    preds = clf.predict(X_train)

    response = {
        'results_overall': {
            'accuracy': accuracy_score(y_train, preds),
            'recall': recall_score(y_train, preds),
            'precision': precision_score(y_train, preds),
            'f1': f1_score(y_train, preds, average="weighted")
        }
    }

    response['results_intents'] = []
    for intent in le.classes_:
        # TODO