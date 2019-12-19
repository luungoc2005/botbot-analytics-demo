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
    response = list(sorted([
        {
            'name': item,
            'count': len(df[df['Intent'] == item])
        }
        for item in intents_list
    ], key=lambda item: -item['count']))

    if top_n > 0:
        response = response[:top_n]

    return jsonify(response)