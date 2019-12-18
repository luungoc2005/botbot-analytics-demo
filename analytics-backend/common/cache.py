import pandas as pd
from ..app import DATA_DIR
from os import path

CACHE = dict()
MAX_CACHE = 5

def get_df_from_file(file_name):
    try:
        if len(CACHE) > MAX_CACHE:
            del CACHE[list(CACHE.keys())[0]]
        if file_name not in CACHE:
            CACHE[file_name] = pd.read_csv(path.join(DATA_DIR, file_name))
        return CACHE[file_name]
    except:
        return None