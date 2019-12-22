from .ignore_lists import MESSAGE_IGNORE_LIST
from underthesea import word_tokenize as vi_word_tokenize
from nltk.tokenize import word_tokenize as en_word_tokenize

import numpy as np

import os

model = None

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

    return tokenize_text_list(text_list)

def tokenize_text_list(input_list):
    if os.environ.get('LANGUAGE', 'en').lower() == 'en':
        text_list = [en_word_tokenize(' '.join(text.split())) for text in input_list]
    else:
        # Vietnamese
        text_list = [vi_word_tokenize(' '.join(text.split())) for text in input_list]
    return text_list

def get_sentence_vectors(text_list):
    vectors = [
        np.average(
            np.array([get_word_vector(word) for word in sentence])
        , axis=0)
        for sentence in text_list
        if len(sentence) > 0
    ]
    return np.array(vectors)

def get_word_vector(word):
    global model
    if model is None:
        # import fasttext
        # if os.environ.get('LANGUAGE', 'en').lower() == 'en':
        #     print('Loading English word vectors')
        #     model = fasttext.load_model('data/cc.en.300.bin')
        # else:
        #     print('Loading Vietnamese word vectors')
        #     model = fasttext.load_model('data/cc.vi.300.bin')

    # return model.get_word_vector(word.replace(' ', '_'))
    
        from pymagnitude import Magnitude

        if os.environ.get('LANGUAGE', 'en').lower() == 'en':
            print('Loading English word vectors')
            model = Magnitude('data/cc.en.300.magnitude', language='en', lazy_loading=20000)
        else:
            print('Loading Vietnamese word vectors')
            model = Magnitude('data/cc.vi.300.magnitude', language='vi', lazy_loading=20000)
        
    return model.query(word.replace(' ', '_'))