from config import DATA_DIR, CACHE_DIR
from common import cache, utils, ignore_lists
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

parser.add_argument("--callback_url", type=str, default='')
parser.add_argument("--sid", type=str, default='')

args = parser.parse_args()

if __name__ == '__main__':
    print(args)

    file_name = args.file_name

    file_path = path.join(DATA_DIR, file_name)

    if not path.exists(CACHE_DIR) or not path.isdir(CACHE_DIR):
        makedirs(CACHE_DIR)
    
    file_hash = cache.get_file_hash(file_path)

    data_cache_path = path.join(CACHE_DIR, f'torchdata_{file_hash}.pt')

    if path.exists(data_cache_path) and path.isfile(data_cache_path):
        train_dataset = torch.load(data_cache_path)
    else:
        train_dataset = DfTrainingDataset(file_path)
        torch.save(train_dataset, data_cache_path)

    clf = LSTMTextClassifier({
        'embedding_size': train_dataset.embedding_size,
        'context_size': train_dataset.context_size,
        'num_classes': train_dataset.num_classes
    }, train_dataset)

    # clf = CNNTextClassifier({
    #     'embedding_size': train_dataset.embedding_size,
    #     'context_size': train_dataset.context_size,
    #     'num_classes': train_dataset.num_classes
    # }, train_dataset)

    cache_path = path.join(CACHE_DIR, f'torchmodel_{file_hash}.pt')

    if path.exists(cache_path) and path.isfile(cache_path):
        clf.load_state_dict(torch.load(cache_path))
    else:
        from pytorch_lightning import Trainer
        from pytorch_lightning.callbacks import EarlyStopping

        early_stop_callback = EarlyStopping(
            monitor='train_loss',
            min_delta=0.00,
            patience=5,
            verbose=False,
            mode='min'
        )

        trainer = Trainer(early_stop_callback=early_stop_callback)
        trainer.fit(clf)

        torch.save(clf.state_dict(), cache_path)

    with torch.no_grad():
        clf.eval()
        preds_proba = torch.softmax(clf(
                train_dataset.X_train, train_dataset.X_contexts
            ),
            dim=-1
        )
        preds_proba = preds_proba.cpu().numpy()
        clf.train()

    preds = np.argmax(preds_proba, axis=-1)

    raw_examples = train_dataset.raw_examples
    raw_labels = train_dataset.raw_labels
    raw_contexts = train_dataset.raw_contexts
    examples_counts = train_dataset.examples_counts
    has_contexts = train_dataset.has_contexts
    X_train = train_dataset.X_train.cpu().numpy()
    y_train = train_dataset.y_train.cpu().numpy()

    examples_counts_keys = list(examples_counts.keys())
    examples_counts_values = list(examples_counts.values())

    max_examples = max(examples_counts_values)
    min_examples = min(examples_counts_values)
    med_examples = statistics.median(examples_counts_values)
    z_scores = np.abs(stats.zscore(examples_counts_values))

    response = {
        'intents_count': len(raw_labels),
        'examples_count': len(raw_examples),
        'stats_overall': {
            # 'accuracy': accuracy_score(y_train, preds),
            'recall': recall_score(y_train, preds, average="micro"),
            'precision': precision_score(y_train, preds, average="micro"),
            'f1': f1_score(y_train, preds, average="micro"),
            'intents_count': len(examples_counts.keys()),
            'examples_count': len(X_train),
            'max_examples': {
                'value': max_examples,
                'intents': [
                    key for key, value in examples_counts.items() 
                    if value == max_examples
                ]
            },
            'min_examples': {
                'value': min_examples,
                'intents': [
                    key for key, value in examples_counts.items() 
                    if value == min_examples
                ]
            },
            'median': med_examples
        }
    }

    results_intents = []

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
        
        thres_recall.append(float(
            recall_score(y_train, th_preds, average="weighted")
        ))
        thres_precision.append(float(
            precision_score(y_train, th_preds, average="weighted")
        ))
        thres_f1.append(float(
            f1_score(y_train, th_preds, average="weighted")
        ))

    response['thresholds_plot'] = {
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

    optimal_thres = threshold_levels[int(np.argmax(thres_f1))]
    response['suggested_threshold'] = float(optimal_thres)

    # by intents

    for class_idx, class_name in enumerate(train_dataset.classes_):
        
        class_mask = (y_train == class_idx)

        preds_class = preds[class_mask]
        y_class = y_train[class_mask]

        # y_class[y_class != class_idx] = -1
        # preds_class[preds_class != class_idx] = -1
        # y_class = np.where(y_class == class_idx, 1, 0)
        # preds_class = np.where(preds_class == class_idx, 1, 0)
        
        intent_results = {
            'name': class_name,
            'accuracy': float(accuracy_score(y_class, preds_class)),
            'correct': int(np.sum(y_class == preds_class)),
            'incorrect': int(np.sum(y_class != preds_class)),
            'total': int(np.sum(class_mask)),
            'problem_examples': []
        }

        results_intents.append(intent_results)

    diff_mask = (y_train != preds)
    diff_indices = np.arange(0, len(y_train))[diff_mask]

    for diff_idx in diff_indices:
        pred_cls = preds[diff_idx]
        gt_cls = y_train[diff_idx]

        results_intents[gt_cls]['problem_examples'].append({
            'text': raw_examples[diff_idx],
            'predicted': train_dataset.classes_[int(pred_cls)],
            'confidence': float(preds_proba[diff_idx][pred_cls])
            # 'ground_truth': f'{train_dataset.classes_[gt_cls]} [{gt_cls}]'
        })
    
    # filter out items without issues
    results_intents = [
        item for item in results_intents
        if item['accuracy'] < 1.0
    ]
    response['results_intents'] = results_intents

    # TODO: could be improved
    overall_intents_plot = {}
    for idx in range(len(X_train)):
        gt_class = y_train[idx]
        pred_class = preds[idx]
        pred_proba = preds_max_proba[idx]

        if gt_class not in overall_intents_plot:
            overall_intents_plot[gt_class] = {
                'correct': 0,
                'incorrect': 0,
                'unclear': 0
            }
        
        if gt_class == pred_class:
            overall_intents_plot[gt_class]['correct'] += 1
        elif pred_proba > optimal_thres:
            overall_intents_plot[gt_class]['incorrect'] += 1
        else:
            overall_intents_plot[gt_class]['unclear'] += 1

    x_labels = ['Correctly predicted', 'Incorrectly predicted', 'Unclear predicted']
    classes_list = list(train_dataset.classes_)
    response['overall_intents_plot'] = {
        'data': [
            {
                'x': classes_list,
                'y': [item[value_name] for item in overall_intents_plot.values()],
                'name': x_labels[idx],
                'type': 'bar'
            }
            for idx, value_name in enumerate(['correct', 'incorrect', 'unclear'])
        ],
        'layout': {
            'displaylogo': False,
            'showlegend': False,
            'barmode': 'stack'
        }
    }

    preds_optimal_th = np.copy(preds)
    preds_optimal_th[preds_max_proba <= optimal_thres] = -1
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
                'type': 'pie'
            }
        ],
        'layout': {
            'displaylogo': False
        }
    }

    # list problems
    problems = []

    # intents with 0 examples
    if 0 in examples_counts_values:
        problems.append({
            'name': 'empty_intents',
            'intents': [
                examples_counts_keys[idx] for idx, count
                in enumerate(examples_counts_values)
                if count == 0
            ]
        }) 

    # unbalanced data:
    if np.sum(z_scores > 3) > 0:
        problems.append({
            'name': 'unbalanced_data',
            'intents': [
                f'{examples_counts_keys[idx]} ({examples_counts_values[idx]} examples)' for idx, score
                in enumerate(z_scores)
                if score > 3
            ]
        })

    # similar intents
    similar_intents = []

    for intent in results_intents:
        total_examples = intent['total']
        if intent['incorrect'] > intent['correct']:
            incorrect_results = [item['predicted'] for item in intent['problem_examples']]
            incorrect_results_set = set(incorrect_results)

            for intent_name in incorrect_results_set:
                result_count = incorrect_results.count(intent_name)

                if result_count >= (total_examples / 3):
                    similar_intents.append({
                        'name': intent['name'],
                        'similar_to': intent_name
                    })
                    break
    
    if len(similar_intents) > 0:
        problems.append({
            'name': 'similar_intents',
            'intents': [
                f'{item["name"]} (similar to "{item["similar_to"]}")'
                for item in similar_intents
            ]
        })

    response['similar_intents'] = similar_intents
    
    response['problems'] = problems

    # print(json.dumps(response, indent=4))
   
    # print('Writing response to cache')
    # with open(cache_path, 'w') as cached_file: 
    #     json.dump(response, cached_file)

    return_response(args, response)