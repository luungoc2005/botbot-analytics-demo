from main_app import DATA_DIR
from routes.history_routes import history_routes_blueprint
from common import cache, utils, vn_utils, ignore_lists, task_scheduler

import numpy as np

from flask import jsonify, escape, request
from werkzeug.exceptions import BadRequest
from os import listdir, path

import string

@history_routes_blueprint.route('/top_similar_words')
def top_similar_words():
    file_name = request.args.get("file", "")
    query_word = request.args.get("word", "")
    top_n = request.args.get("top_n", "-1")

    sid = request.args.get("sid", "")

    file_path = path.join(DATA_DIR, escape(file_name))
    if not path.exists(file_path) or not path.isfile(file_path):
        return BadRequest('File not found')

    if len(query_word) == 0:
        return BadRequest('No word specified')

    return jsonify({
        'task_id': task_scheduler.run_task('tasks.top_similar_words', [
            '--file_name', file_name,
            '--query_word', query_word,
            '--top_n', top_n
        ], request.url_root, sid)
    })
    # df = cache.get_df_from_file(file_name)
    # texts = utils.get_text_list_from_df(df)

    # words = []
    # for sent in texts:
    #     for word in sent:
    #         word = word.lower().strip()
    #         word = vn_utils.remove_tone_marks(word)

    #         words.append(word)

    # words = list(set(words))
    # vectors = [utils.get_word_vector(word) for word in words]
    # query_vector = utils.get_word_vector(query_word)

    # similarity = np.dot(query_vector, np.vstack(vectors).T)
    # indices = np.argsort(similarity)[::-1]

    # if top_n > 0:
    #     indices = indices[:top_n]

    # response = [words[idx] for idx in indices]

    # return jsonify(response)