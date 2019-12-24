import pandas as pd
from config import DATA_DIR
from os import path
import hashlib

CACHE = dict()
MAX_CACHE = 5

def get_file_hash(file_name):
    BUF_SIZE = 65536

    md5 = hashlib.md5()

    with open(file_name, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            md5.update(data)
    
    return str(md5.hexdigest())

def get_df_from_file(file_name):
    try:
        if len(CACHE) > MAX_CACHE:
            del CACHE[list(CACHE.keys())[0]]
        if file_name not in CACHE:
            file_path = path.join(DATA_DIR, file_name)
            print(file_path)
            df = pd.read_csv(file_path)
            df['MessageReceived'] = pd.to_datetime(df['MessageReceived'])
            df = df.set_index(pd.DatetimeIndex(df['MessageReceived']))

            CACHE[file_name] = df
            
        return CACHE[file_name]
    except:
        return None