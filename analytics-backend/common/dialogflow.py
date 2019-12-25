from zipfile import ZipFile
from config import CACHE_DIR, DATA_DIR
from os import path, makedirs
from common.cache import get_file_hash
import json
import re

# crudely convert Dialogflow export format to this dashboard's format
def load_dialogflow_archive(file_path):
    if not path.exists(file_path) or not path.isfile(file_path):
        raise ValueError('File does not exist')

    file_hash = get_file_hash(file_path)
    cache_path = path.join(CACHE_DIR, f'dfdata_{file_hash}')
    if not path.exists(cache_path) or not path.isdir(cache_path):
        makedirs(cache_path)

    INTENTS_PREFIX = 'intents/'
    ret_val = []

    with ZipFile(file_path) as zip_file:
        content_list = zip_file.namelist()
        intent_files = [
            item for item in content_list 
            if item.startswith(INTENTS_PREFIX)
            and re.search(r"_usersays_[a-z]{2}.json$", item) is not None
        ]
        # intent_names = [
        #     item[len(INTENTS_PREFIX):].split('_usersays')[0]
        #     for item in intent_files
        # ]
        # print(intent_names)
        for intent_file in intent_files:
            intent_main_file_name = ''.join(intent_file.split('_usersays')[:-1])
            intent_name = intent_main_file_name[len(INTENTS_PREFIX):]

            intent_main_file_name = intent_main_file_name + '.json'

            if intent_main_file_name in content_list:
                print(f'Reading {intent_main_file_name} and {intent_file}')
                intent_data = {
                    'name': intent_name,
                    'usersays': [],
                    'contexts': []
                }
                with zip_file.open(intent_main_file_name) as intent_file_fp:
                    intent_json = json.load(intent_file_fp)
                    intent_data['contexts'] = intent_json.get('contexts', [])

                with zip_file.open(intent_file) as usersays_file_fp:
                    usersays_json = json.load(usersays_file_fp)
                    usersays = []
                    for usersay_item in usersays_json:
                        item_data = usersay_item.get('data', [])
                        if len(item_data) == 0:
                            break
                            
                        item_data = [item.get('text', '') for item in item_data]
                        item_data = ''.join(item_data)
                        item_data = ' '.join(item_data.split()) # remove redundant spaces

                        usersays.append(item_data)
                    intent_data['usersays'] = usersays
            
            ret_val.append(intent_data)
        # print(json.dumps(ret_val, indent=4))
        return ret_val
                # intent_data['usersays'] = 


if __name__ == "__main__":
    # sanity check
    test_file = 'NTUC-HelpDesk-Bot.zip'
    test_path = path.join(DATA_DIR, test_file)
    print(test_path)
    load_dialogflow_archive(test_path)