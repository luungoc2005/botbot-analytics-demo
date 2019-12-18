from ..app import app, DATA_DIR
from ..common import cache, utils, ignore_lists
from flask import jsonify, escape, request
from werkzeug.exceptions import BadRequest
from os import listdir, path

import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE

from typing import List, Tuple

def visualize_matrix(X: np.array) -> Tuple[np.array, np.array]:
    y_pred = DBSCAN().fit_predict(X)

    print("Analysing with t-SNE...")
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    x_min, x_max = np.min(X_tsne, 0), np.max(X_tsne, 0)
    X_tsne = (X_tsne - x_min) / (x_max - x_min)

    return X_tsne, y_pred

@app.route('/clustering_visualize')
def clustering_visualize():
    file_name = request.args.get("file", "")
    only_fallback = request.args.get("only_fallback", "")

    if only_fallback.lower() in ['1', 'true']:
        only_fallback = True

    file_path = path.join(DATA_DIR, escape(file_name))
    if not path.exists(file_path):
        return BadRequest('File not found')

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
                'mode': 'markers',
                'type': 'scatter',
                'name': f'Cluster #{item_group}',
                'text': [],
                'marker': { 'size': 12 }
            }
            if item_group == -1:
                viz_groups[item_group]['name'] = '(noise)'
                viz_groups[item_group]['marker']['color'] = 'black'
        viz_groups[item_group]['x'].append(float(X_tsne[i, 0]))
        viz_groups[item_group]['y'].append(float(X_tsne[i, 1]))
        viz_groups[item_group]['text'].append(' '.join(text_list[i]))
    
    return jsonify({
        'data': viz_groups.values(),
        'layout': {
            'xaxis': {
                'range': [0, 1]
            },
            'yaxis': {
                'range': [0, 1]
            },
            'title':'Grouped messages from users'
        }
    })