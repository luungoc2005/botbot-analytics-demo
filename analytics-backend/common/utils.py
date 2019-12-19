from .ignore_lists import MESSAGE_IGNORE_LIST
from underthesea import word_tokenize

import numpy as np
import fasttext

model = fasttext.load_model('data/cc.vi.300.bin')

def get_text_list_from_df(df, is_in=None, is_not_in=None):
    df_tmp = df.copy()
    if is_in is not None:
        df_tmp = df_tmp[df_tmp['Intent'].isin(is_in)]
    if is_not_in is not None:
        df_tmp = df_tmp[~df_tmp['Intent'].isin(is_not_in)]
    text_list = df_tmp['User Message']
    text_list = [
        text.strip() for text in text_list 
        if text not in MESSAGE_IGNORE_LIST
        and len(text.strip()) > 0
    ]
    text_list = [word_tokenize(' '.join(text.split())) for text in text_list]
    return text_list

def get_sentence_vectors(text_list):
    vectors = [
        np.average(
            np.array([model.get_word_vector(word) for word in sentence])
        , axis=0)
        for sentence in text_list
        if len(sentence) > 0
    ]
    return np.array(vectors)