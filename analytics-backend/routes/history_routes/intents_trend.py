from config import DATA_DIR
from routes.history_routes import history_routes_blueprint
from common import cache, utils, vn_utils, ignore_lists

import pandas as pd
import numpy as np

from flask import jsonify, escape, request
from werkzeug.exceptions import BadRequest
from os import listdir, path

import string

@history_routes_blueprint.route('/intents_trend')
def intents_trend():
    file_name = request.args.get("file", "")
    # only = request.args.get("only", "").split(",")
    intents = request.args.get("intents", "").split(",")
    period = request.args.get("period", "D").upper()

    file_path = path.join(DATA_DIR, escape(file_name))
    if not path.exists(file_path) or not path.isfile(file_path):
        return BadRequest('File not found')

    if not period in ['D', 'M']:
        return BadRequest('Invalid frequency in request')

    intents = [intent for intent in intents if len(intent.strip()) > 0]
    if len(intents) == 0:
        return BadRequest('No intent specified in request')

    df = cache.get_df_from_file(file_name)
    
    data = []

    for intent in intents:
        intent = intent.lower().strip()

        tmp_df = df.copy()
        tmp_df['Intent'] = tmp_df.apply(lambda x: x['Intent'].lower().strip(), axis=1)
        tmp_df = tmp_df[tmp_df['Intent'] == intent]
        
        intent_result = tmp_df.groupby(tmp_df.index.to_period(period)).count()[['Intent']]
        intent_result.index = pd.to_datetime(intent_result.index.to_timestamp())
        intent_result.index = intent_result.apply(lambda x: x.index.strftime("%Y-%m-%d %H:%M:%S"))

        data_series = {
            'x': [item[0] for item in intent_result.index],
            'y': list(intent_result['Intent']),
            'type': 'scatter',
            'name': f'"{intent}" triggers'
        }
        
        data.append(data_series)

    return jsonify({
        'data': data,
        'layout': {
            'title': 'Triggers over time'
        }
    })