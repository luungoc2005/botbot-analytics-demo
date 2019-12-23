from config import DATA_DIR
from common import cache, utils, ignore_lists

from os import listdir, path

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str, default='')

parser.add_argument("--callback_url", type=str, default='')
parser.add_argument("--sid", type=str, default='')

args = parser.parse_args()

if __name__ == '__main__':
    print(args)

    file_name = args.file_name

    file_path = path.join(DATA_DIR, file_name)

    with open(file_path, 'r') as input_file:
        training_file = json.load(input_file)

    # extract raw intents and labels from file. Needs upgrade
    raw_examples = []
    raw_labels = []
    examples_counts = {}

    for intent_idx, intent in enumerate(training_file):
        for usersay in intent['usersays']:
            raw_labels.append(intent['name'])
            raw_examples.append(usersay)
        examples_counts[intent_idx] = len(intent['usersays'])

    raw_exampes_tokens = utils.tokenize_text_list(raw_examples)
    
    le = LabelEncoder()
    X_train = utils.get_sentence_vectors(raw_exampes_tokens)
    y_train = le.fit_transform(raw_labels)

    clf = MLPClassifier(
        hidden_layer_sizes=(50,), 
        random_state=1,
        batch_size=min(32, len(X_train)),
        max_iter=5,
        verbose=True)

    clf.fit(X_train, y_train)

    preds_proba = clf.predict_proba(X_train)

    preds = np.argmax(preds_proba, axis=-1)

    response = {
        'intents_count': len(raw_labels),
        'examples_count': len(raw_examples),
        'results_overall': {
            # 'accuracy': accuracy_score(y_train, preds),
            'recall': recall_score(y_train, preds, average="micro"),
            'precision': precision_score(y_train, preds, average="micro"),
            'f1': f1_score(y_train, preds, average="micro")
        }
    }

    results_intents = []

    for class_idx, class_name in enumerate(le.classes_):
        class_mask = (y_train == class_idx)

        preds_class = preds[class_mask]
        y_class = y_train[class_mask]
        
        y_class[y_class != class_idx] = -1
        preds_class[preds_class != class_idx] = -1

        intent_results = {
            'name': class_name,
            'accuracy': accuracy_score(y_class, preds_class),
            'recall': recall_score(y_class, preds_class, pos_label=class_idx),
            'precision': precision_score(y_class, preds_class, pos_label=class_idx),
            'f1': f1_score(y_class, preds_class, pos_label=class_idx),
            'problem_examples': []
        }

        results_intents.append(intent_results)

    diff_mask = (y_train != preds)
    diff_indices = np.arange(0, len(y_train))[diff_mask]

    for diff_idx in diff_indices:
        pred_cls = preds[diff_idx]
        gt_cls = y_train[diff_idx]

        results_intents[gt_cls]['problem_examples'].append({
            'example': raw_examples[diff_idx],
            'predicted': le.classes_[pred_cls],
            'confidence': preds_proba[diff_idx][pred_cls]
            # 'ground_truth': le.classes_[gt_cls]
        })
    
    response['results_intents'] = results_intents

    # thresholds
    results_thresholds = {}
    preds_max_proba = np.amax(preds_proba, axis=-1)

    threshold_levels = np.arange(0, 1, .01).tolist()
    thres_precision = []
    thres_recall = []
    thres_f1 = []
    for threshold_level in threshold_levels:
        th_mask = (preds_max_proba <= threshold_level)
        th_preds = np.copy(preds)
        th_preds[th_mask] = -1
        
        thres_recall.append(recall_score(y_train, th_preds, average="micro"))
        thres_precision.append(precision_score(y_train, th_preds, average="micro"))
        thres_f1.append(f1_score(y_train, th_preds, average="micro"))

    response['thresholds'] = {
        'data': [
            {
                'x': threshold_levels,
                'y': thres_precision,
                'mode': 'lines',
                'name': 'Precision'
            },
            {
                'x': threshold_levels,
                'y': thres_recall,
                'mode': 'lines',
                'name': 'Recall'
            }
        ],
        'layout': {
            'displaylogo': False
        }
    }

    optimal_thres = np.argmax(thres_f1)
    response['suggested_thres'] = float(optimal_thres)
    preds_optimal_th = np.copy(preds)
    preds_optimal_th[preds_max_proba <= preds_optimal_th] = -1

    response['overall_plot'] = {
        'data': [
            {
                'values': [
                    int(np.sum(preds_optimal_th == y_train)),
                    int(np.sum(preds_optimal_th != y_train)),
                    int(np.sum(preds_optimal_th == -1))
                ],
                'labels': [
                    'Correctly predicted',
                    'Incorrectly predicted',
                    'Unclear predicted'
                ],
                'hoverinfo': 'label+percent',
            }
        ],
        'layout': {
            'displaylogo': False
        }
    }

    # list problems
    # TODO

    print(json.dumps(response, indent=4))
    if args.callback_url.strip() != '':
        # from urllib import request, parse
        import requests

        print('Sending POST request to', args.callback_url)
        # data = json.dumps(response).encode('utf8')
        request_obj = requests.post(
            args.callback_url,
            data=json.dumps(response)
        )