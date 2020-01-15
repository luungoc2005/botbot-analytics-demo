
from os import path, makedirs, listdir
import subprocess
import argparse
import h5py
import numpy as np
from nltk import sent_tokenize
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--base_path', type=str, default='')
parser.add_argument('--output_path', type=str, default='')
parser.add_argument('--model_prefix', type=str, default='en')
parser.add_argument('--dataset_type', type=str, default='train')
parser.add_argument('--max_sentence_length', type=int, default=128)

args = parser.parse_args()

buffer = []

if __name__ == '__main__':
    BASE_PATH = args.base_path
    OUTPUT_PATH = args.output_path
    assert args.dataset_type in ['train', 'test']

    paths = []

    def load_folder(folder_path, filter_txt=True):
        paths.extend([
            path.join(folder_path, filename)
            for filename in listdir(folder_path)
            if not filter_txt or filename.lower().endswith('txt')
        ])

    if args.dataset_type == 'train':
        paths = [
            # path.join(BASE_PATH, 'wikitext2/wiki.train.tokens'),
            path.join(BASE_PATH, 'wikitext103raw/wiki.train.raw')
        ]

        # load_folder(path.join(BASE_PATH, 'bookcorpus'))
        load_folder(path.join(
            BASE_PATH, '1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled'), filter_txt=False)
        load_folder(path.join(BASE_PATH, 'stories_corpus'))
    else:
        paths = [
            path.join(BASE_PATH, 'wikitext103raw/wiki.test.raw')
        ]

        load_folder(path.join(
            BASE_PATH, '1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled'), filter_txt=False)

    if not path.isdir(OUTPUT_PATH):
        makedirs(OUTPUT_PATH)

    TOKENIZER_PATH = path.join(OUTPUT_PATH, 'sentencepiece')
    
    # import sentencepiece as spm
    # sp = spm.SentencePieceProcessor()
    # sp_model_path = path.join(TOKENIZER_PATH, f'{args.model_prefix}.model')
    # sp.Load(sp_model_path)

    from tokenizers import BertWordPieceTokenizer
    tokenizer_path = path.join(TOKENIZER_PATH, f'{args.model_prefix}-vocab.txt')
    tokenizer = BertWordPieceTokenizer(tokenizer_path,
        add_special_tokens=True,
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=False,
        lowercase=False
    )

    data_path = path.join(OUTPUT_PATH, f'data_{args.dataset_type}.h5')

    REF_CHUNK_SIZE = 320000

    with h5py.File(data_path, 'w') as hf:
        def process_buffer():
            global buffer
            chunk_size = len(buffer)
            x_buffer = np.zeros((chunk_size, args.max_sentence_length))
            length_buffer = np.zeros((chunk_size,), dtype=np.int64)

            for ix, sentence in enumerate(buffer):
                # tokens = sp.EncodeAsIds(sentence)
                tokens = tokenizer.encode(sentence).ids
                true_length = len(tokens)

                if true_length > args.max_sentence_length:
                    tokens = tokens[:args.max_sentence_length]
                    true_length = args.max_sentence_length

                x_buffer[ix,:len(tokens)] = tokens
                length_buffer[ix] = true_length

            if not 'tokens' in hf:
                hf.create_dataset('tokens', data=x_buffer, \
                    compression="gzip", chunks=True, maxshape=(None,args.max_sentence_length))
            else:
                hf['tokens'].resize((hf['tokens'].shape[0] + chunk_size), axis=0)
                hf['tokens'][-chunk_size:] = x_buffer

            if not 'lengths' in hf:
                hf.create_dataset('lengths', data=length_buffer, \
                    compression="gzip", chunks=True, maxshape=(None,))
            else:
                hf['lengths'].resize((hf['lengths'].shape[0] + chunk_size), axis=0)
                hf['lengths'][-chunk_size:] = length_buffer

            buffer = []

        for file_name in paths:
            print(f'Processing {file_name}')

            with open(file_name, 'r', encoding='utf-8') as txt_file:
                for line in tqdm(txt_file):
                    stripped = line.strip()
                    if stripped == '' or stripped.startswith('=') or stripped.startswith('~~'):
                        continue

                    buffer.extend([sent_line.strip() for sent_line in sent_tokenize(line)])

                    if len(buffer) > REF_CHUNK_SIZE:
                        process_buffer()