from .ignore_lists import MESSAGE_IGNORE_LIST
from underthesea import word_tokenize as vi_word_tokenize
from nltk.tokenize import word_tokenize as en_word_tokenize
from config import CACHE_DIR
from os import path

import numpy as np

import os
import json

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

def get_sentence_vectors(text_list, use_cache=True):
    # hashing for cache
    cache_file = None
    if use_cache:
        import hashlib
        md5 = hashlib.md5()
        data_str = json.dumps(text_list)
        md5.update(data_str.encode('utf-8'))

        cache_file = path.join(CACHE_DIR, f'vectors_{str(md5.hexdigest())}.npy')
        print(cache_file)

        if path.exists(cache_file) and path.isfile(cache_file):
            print('Vector cache hit')
            return np.load(cache_file)

    vectors = np.array([
        np.average(
            np.array(get_word_vector([word.replace(' ', '') for word in sentence]))
        , axis=0)
        for sentence in text_list
        if len(sentence) > 0
    ])

    if cache_file is not None:
        np.save(cache_file, vectors)

    return vectors

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
        
        print('Loading completed')

    return model.query(word)

def get_sentence_vectors_full(text_list, length=80):
    """
    get non-averaged sentence vectors
    """
    num_sents = len(text_list)
    ret_val = None
    
    for sent_ix in range(num_sents):
        sent_vec = np.array(get_word_vector([word.replace(' ', '') 
            for word in text_list[sent_ix]]))
        
        sent_length, emb_size = sent_vec.shape

        if ret_val is None:
            ret_val = np.zeros((num_sents, length, emb_size))
        
        vec_length = min(sent_length, length)

        ret_val[sent_ix,:vec_length] = sent_vec[:vec_length]

    return ret_val
