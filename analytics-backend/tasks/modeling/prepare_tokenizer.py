from os import path, makedirs, listdir
import subprocess
import argparse
import h5py
from nltk import sent_tokenize

parser = argparse.ArgumentParser()
parser.add_argument('--base_path', type=str, default='')
parser.add_argument('--output_path', type=str, default='')
parser.add_argument('--model_prefix', type=str, default='en')
parser.add_argument('--vocab_size', type=int, default=12000)
parser.add_argument('--max_sentence_length', type=int, default=128)

args = parser.parse_args()

if __name__ == '__main__':
    BASE_PATH = args.base_path
    OUTPUT_PATH = args.output_path

    paths = [
        # path.join(BASE_PATH, 'data/wikitext2/wiki.train.tokens'),
        path.join(BASE_PATH, 'wikitext103raw/wiki.train.raw')
    ]

    def load_folder(folder_path, filter_txt=True):
        paths.extend([
            path.join(folder_path, filename)
            for filename in listdir(folder_path)
            if not filter_txt or filename.lower().endswith('txt')
        ])
    # load_folder(path.join(BASE_PATH, 'bookcorpus'))
    # load_folder(path.join(
    #     BASE_PATH, '1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled'), filter_txt=False)
    load_folder(path.join(BASE_PATH, 'stories_corpus'))

    if not path.isdir(OUTPUT_PATH):
        makedirs(OUTPUT_PATH)

    TOKENIZER_PATH = path.join(OUTPUT_PATH, 'sentencepiece')
    
    if not path.isdir(TOKENIZER_PATH):
        makedirs(TOKENIZER_PATH)

    # import sentencepiece as spm
    # spm.SentencePieceTrainer.Train(' '.join([
    #     f'--input={",".join(paths)}',
    #     f'--model_prefix={args.model_prefix}',
    #     f'--vocab_size={args.vocab_size}',
    #     f'--character_coverage={1.0}',
    #     f'--model_type=bpe'
    # ]))
    from tokenizers import BertWordPieceTokenizer
    
    tokenizer = BertWordPieceTokenizer(
        add_special_tokens=True,
        clean_text=True,
        handle_chinese_chars=True,
        strip_accents=False,
        lowercase=False
    )
    # MAX_FILES = 300
    # if len(paths) > MAX_FILES:
    #     # only take 100 files
    #     paths = paths[:MAX_FILES]

    tokenizer.train(paths, vocab_size=args.vocab_size, show_progress=False)
    tokenizer.save(TOKENIZER_PATH, args.model_prefix)