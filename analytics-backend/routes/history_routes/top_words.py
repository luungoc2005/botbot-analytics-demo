from main_app import DATA_DIR
from routes.history_routes import history_routes_blueprint
from common import cache, utils, vn_utils, ignore_lists

from flask import jsonify, escape, request
from werkzeug.exceptions import BadRequest
from os import listdir, path

import string

@history_routes_blueprint.route('/top_words')
def top_words():
    file_name = request.args.get("file", "")
    only = request.args.get("only", "").split(',')
    top_n = int(request.args.get("top_n", "-1"))

    file_path = path.join(DATA_DIR, escape(file_name))
    if not path.exists(file_path) or not path.isfile(file_path):
        return BadRequest('File not found')

    df = cache.get_df_from_file(file_name)
    texts = utils.get_text_list_from_df(df, is_in=only)

    wcount = {}
    for sent in texts:
        for word in sent:
            word = word.strip().lower()

            if word in string.punctuation:
                continue

            word = vn_utils.remove_tone_marks(word)
            if word not in wcount:
                wcount[word] = 1
            else:
                wcount[word] += 1
    
    response = list(sorted([{
            'word': key,
            'count': value
        }
        for key, value in wcount.items()], 
        key=lambda item: -item['count'])
    )

    if top_n > 0:
        response = response[:top_n]

    return jsonify(response)