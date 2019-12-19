from ..app import app, DATA_DIR
from ..common import cache, utils, ignore_lists

from flask import jsonify, escape, request
from werkzeug.exceptions import BadRequest
from os import listdir, path

@app.route('/intents_list')
def intents_list():
    file_name = request.args.get("file", "")
    file_path = path.join(DATA_DIR, escape(file_name))
    if not path.exists(file_path) or not path.isfile(file_path):
        return BadRequest('File not found')

    df = cache.get_df_from_file(file_name)
    return jsonify(list(sorted(set(df['Intent']))))