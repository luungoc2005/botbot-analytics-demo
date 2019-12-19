from ..app import app, DATA_DIR
from ..common import cache, utils, vn_utils, ignore_lists

import pandas as pd
import numpy as np

from flask import jsonify, escape, request
from werkzeug.exceptions import BadRequest
from os import listdir, path

import string

def count_occurences(source_str, target_str):
    source_str = source_str.lower().strip()
    source_str = vn_utils.remove_tone_marks(source_str)
    return source_str.count(target_str)

@app.route('/words_trend')
def words_trend():
    file_name = request.args.get("file", "")
    # only = request.args.get("only", "").split(",")
    words = request.args.get("words", "").split(",")
    period = request.args.get("period", "D").upper()

    file_path = path.join(DATA_DIR, escape(file_name))
    if not path.exists(file_path) or not path.isfile(file_path):
        return BadRequest('File not found')

    if not period in ['D', 'M']:
        return BadRequest('Invalid frequency in request')

    words = [word for word in words if len(word.strip()) > 0]
    if len(words) == 0:
        return BadRequest('No word specified in request')

    df = cache.get_df_from_file(file_name)
    
    data = []

    for word in words:
        word = word.lower().strip()
        word = vn_utils.remove_tone_marks(word)

        tmp_df = df.copy()
        tmp_df['Count'] = tmp_df.apply(lambda x: count_occurences(x['User Message'], word), axis=1)
        
        word_result = tmp_df.groupby(df.index.to_period(period)).sum()[['Count']]
        word_result.index = pd.to_datetime(word_result.index.to_timestamp())
        word_result.index = word_result.apply(lambda x: x.index.strftime("%Y-%m-%d %H:%M:%S"))

        data_series = {
            'x': [item[0] for item in word_result.index],
            'y': list(word_result['Count']),
            'type': 'scatter',
            'name': f'"{word}" mentions'
        }
        
        data.append(data_series)

    return jsonify({
        'data': data,
        'layout': {
            'title': 'Mentions over time'
        }
    })