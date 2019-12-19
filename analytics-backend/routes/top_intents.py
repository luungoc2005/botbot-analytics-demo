from ..app import app, DATA_DIR
from ..common import cache, utils, ignore_lists

from flask import jsonify, escape, request
from werkzeug.exceptions import BadRequest
from os import listdir, path

@app.route('/top_intents')
def top_intents():
    file_name = request.args.get("file", "")
    only = request.args.get("only", "").split(',')
    top_n = int(request.args.get("top_n", "-1"))

    file_path = path.join(DATA_DIR, escape(file_name))
    if not path.exists(file_path) or not path.isfile(file_path):
        return BadRequest('File not found')

    df = cache.get_df_from_file(file_name)
    intents_list = list(set(df['Intent']))
    intents_list = [item for item in intents_list if item in only]
    intents_list = list(sorted([
        {
            'name': item,
            'count': len(df[df['Intent'] == item])
        }
        for item in intents_list
    ], key=lambda item: -item['count']))

    # plot a pie chart


    if top_n > 0:
        intents_list = intents_list[:top_n]

    return jsonify({
        'list': intents_list,
        'plot': {
            'data': [{
                'values': [item.get('count', 0) for item in intents_list],
                'labels': [item.get('name', '') for item in intents_list],
                'type': 'pie',
                'textinfo': 'label+percent',
                'textposition': 'outside',
                'automargin': True
            }],
            'layout': {
                'showlegend': False
            }
        }
    })