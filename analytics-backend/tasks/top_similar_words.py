
from config import DATA_DIR
from common import cache, utils, vn_utils, ignore_lists

import numpy as np

from os import listdir, path

import string

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str, default='')
parser.add_argument('--query_word', type=str, default='')
parser.add_argument('--top_n', type=str, default='-1')

parser.add_argument("--callback_url", type=str, default='')
parser.add_argument("--sid", type=str, default='')

args = parser.parse_args()

if __name__ == '__main__':
    file_name = args.file_name
    query_word = args.query_word
    top_n = int(args.top_n)

    df = cache.get_df_from_file(file_name)
    texts = utils.get_text_list_from_df(df)

    words = []
    for sent in texts:
        for word in sent:
            word = word.lower().strip()
            word = vn_utils.remove_tone_marks(word)

            words.append(word)

    words = list(set(words))
    vectors = [utils.get_word_vector(word) for word in words]
    query_vector = utils.get_word_vector(query_word)

    similarity = np.dot(query_vector, np.vstack(vectors).T)
    indices = np.argsort(similarity)[::-1]

    if top_n > 0:
        if indices[0] == query_word:
            indices = indices[1:top_n + 1]
        else:
            indices = indices[:top_n]

    response = [words[idx] for idx in indices]

    # return jsonify(response)
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