from common import cache, utils, ignore_lists
from os import listdir, path
from config import DATA_DIR

import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

from typing import List, Tuple

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str, default='')
parser.add_argument('--only_fallback', type=str, default='')

parser.add_argument("--callback_url", type=str, default='')
parser.add_argument("--sid", type=str, default='')

args = parser.parse_args()

def visualize_matrix(X: np.array) -> Tuple[np.array, np.array]:
    y_pred = DBSCAN().fit_predict(X)

    print("Analysing with t-SNE...")
    tsne = TSNE(n_components=3)
    X_tsne = tsne.fit_transform(X)
    x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
    X_tsne = (X_tsne - x_min) / (x_max - x_min)

    return X_tsne, y_pred

if __name__ == '__main__':
    print(args)

    file_name = args.file_name
    only_fallback = args.only_fallback
    only_fallback = only_fallback.lower() in ['1', 'true']

    # file_path = path.join(DATA_DIR, file_name)

    df = cache.get_df_from_file(file_name)

    if only_fallback:
        text_list = utils.get_text_list_from_df(df, is_in=ignore_lists.FALLBACK_INTENTS_LIST)
    else:
        text_list = utils.get_text_list_from_df(df)

    X = utils.get_sentence_vectors(text_list)

    X_tsne, y_pred = visualize_matrix(X)

    viz_groups = {}

    for i in range(X_tsne.shape[0]):
        # item_value = {
        #     'text': ' '.join(text_list[i]),
        #     'x': float(X_tsne[i, 0]),
        #     'y': float(X_tsne[i, 1]),
        # }
        # if y_pred is not None:
        #     item_value['group'] = int(y_pred[i])
        # result.append(item_value)
        item_group = int(y_pred[i])
        if item_group not in viz_groups:
            viz_groups[item_group] = {
                'x': [],
                'y': [],
                'z': [],
                'mode': 'markers',
                'type': 'scatter3d',
                'name': f'Cluster #{item_group}',
                'text': [],
                'marker': { 'size': 12, 'opacity': 0.8 }
            }
            if item_group == -1:
                viz_groups[item_group]['name'] = '(noise)'
                viz_groups[item_group]['marker']['color'] = 'black'
        viz_groups[item_group]['x'].append(float(X_tsne[i, 0]))
        viz_groups[item_group]['y'].append(float(X_tsne[i, 1]))
        viz_groups[item_group]['z'].append(float(X_tsne[i, 2]))
        viz_groups[item_group]['text'].append(' '.join(text_list[i]))
    
    response = {
        'data': list(viz_groups.values()),
        'layout': {
            'xaxis': {
                'range': [0, 1]
            },
            'yaxis': {
                'range': [0, 1]
            },
            'zaxis': {
                'range': [0, 1]
            },
            'title':'Grouped messages from users'
        }
    }

    if args.callback_url.strip() != '':
        # from urllib import request, parse
        import requests
        import json

        print('Sending POST request to', args.callback_url)
        # data = json.dumps(response).encode('utf8')
        request_obj = requests.post(
            args.callback_url,
            data=json.dumps(response)
        )
        # request.urlopen(request_obj)