import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import math

from torch.utils.data import DataLoader, Dataset
from os import path, getcwd

import pytorch_lightning as pl

import h5py

train_dataset_path = path.join(getcwd(), 'tasks/modeling/data/data_train.h5')
test_dataset_path = path.join(getcwd(), 'tasks/modeling/data/data_test.h5')
CHECKPOINT_PATH = '/media/luungoc2005/Data/Projects/botbot-analytics-demo/checkpoints'
VOCAB_PATH = '/home/luungoc2005/Documents/botbot-analytics-demo/analytics-backend/tasks/modeling/data/sentencepiece/en-vocab.txt'
BATCH_SIZE = 120
NUM_WORKERS = 7

MAX_SEQUENCE_LENGTH = 128

tokenizer = None

class HDF5Dataset(Dataset):

    def __init__(self, 
        h5_file_path, 
        dataset_key="tokens", 
        dataset_length_key="lengths", 
        override_length=None,
        max_sequence_length=MAX_SEQUENCE_LENGTH
    ):
        super(HDF5Dataset, self).__init__()

        self.h5_file_path = h5_file_path
        self.h5_file = None
        self.dataset_key = dataset_key
        self.dataset_length_key = dataset_length_key
        self.override_length = override_length
        self.max_sequence_length = max_sequence_length

        with h5py.File(self.h5_file_path, 'r') as h5_file:
            self.dataset_length = h5_file[self.dataset_key].shape[0]

    def _get_dataset(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_file_path, 'r')
        return self.h5_file

    def __len__(self):
        # a hack for pytorch-lightning model checkpoint
        return self.dataset_length if self.override_length is None else self.override_length

    def __getitem__(self, index):
        rand_idx = random.randint(0, self.dataset_length - 1)
        tokens = torch.from_numpy(self._get_dataset()[self.dataset_key][rand_idx]).long()
        length = torch.LongTensor([self._get_dataset()[self.dataset_length_key][rand_idx]])

        if tokens.size(0) > self.max_sequence_length:
            tokens = tokens[:self.max_sequence_length]

        return (tokens, length)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    # torch.autograd.set_detect_anomaly(True)

    from tokenizers import BertWordPieceTokenizer

    if path.isfile(VOCAB_PATH):
        tokenizer = BertWordPieceTokenizer(VOCAB_PATH,
            add_special_tokens=True,
            clean_text=True,
            handle_chinese_chars=True,
            strip_accents=False,
            lowercase=False
        )

    def get_whole_word_mask(vocab_size):
        connect_sign = tokenizer._parameters['wordpieces_prefix']
        def is_beginning_of_word(i):
            # TODO: special tokens?
            try:
                return tokenizer.id_to_token(i).startswith(connect_sign)
            except:
                return False
        mask_whole_words = torch.ByteTensor(list(
            map(is_beginning_of_word, range(vocab_size))
        ))
        return mask_whole_words

    from model_lstm import LSTM_LM, LMClassifierHead, LMGeneratorHead

    class HParams(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

        def __init__(self, dct):
            self.dict = dct
            for key, value in dct.items():
                if hasattr(value, 'keys'):
                    value = HParams(value)
                self[key] = value

    class LMAdversarialModel(pl.LightningModule):

        def __init__(self, hparams):
            super(LMAdversarialModel, self).__init__()
            self.hparams = hparams

            self.generator_lm = LSTM_LM(hparams['generator_lm'])
            self.discriminator_lm = LSTM_LM(hparams['discriminator_lm'])

            self.generator_head = LMGeneratorHead(hparams['generator_head'])
            self.discriminator_head = LMClassifierHead(hparams['discriminator_head'])
            self.discriminator_loss_delta = hparams['discriminator_loss_delta']

            self.generator_lm.embedding.weight = self.discriminator_lm.embedding.weight
            self.discriminator_lm.embedding_linear.weight = self.discriminator_lm.embedding_linear.weight

            self.generator_head.decoder.weight = self.generator_lm.embedding.weight

            # setup whole word masking
            self.mask_whole_words = get_whole_word_mask(
                self.generator_lm.embedding.weight.size(0)
            ).cuda()
            self._whole_words_cache_for_batch_size = {}

        def _generate_mask(self, tokens, input_lengths):
            batch_size = tokens.size(0)
            length_mask = torch.arange(tokens.size(1)).unsqueeze(0).cuda() < input_lengths.unsqueeze(1)

            word_cont_mask = self._whole_words_cache_for_batch_size.get(batch_size)
            if word_cont_mask is None:
                word_cont_mask = self.mask_whole_words.\
                    expand(tokens.size(0), -1)
                self._whole_words_cache_for_batch_size[batch_size] = word_cont_mask
            
            mask_positions = (torch.cuda.FloatTensor(*tokens.size()).uniform_() < .15) & length_mask

            word_fill_mask = word_cont_mask.gather(1, tokens)
            word_fill_mask[:,:word_fill_mask.size(1) - 1] = word_fill_mask[:,1:]
            word_fill_mask[:,-1] = False

            mask_positions = (mask_positions.bool() | word_fill_mask.bool()) & length_mask.bool()
            
            return mask_positions, length_mask

        def forward(self, tokens):
            return self.discriminator_lm(tokens)

        def training_step(self, batch, batch_idx):
            x, x_lengths = batch
            x_lengths = x_lengths.squeeze(1)

            # indices to mask out
            mask_positions, length_mask = self._generate_mask(x, x_lengths)

            x_generator = x.clone()
            x_generator[mask_positions] = self.generator_lm.vocab_size - 1

            # generator
            x_generator = self.generator_lm(x_generator, input_lengths=x_lengths)

            # if sample_size == 0:
            # loss on the whole batch
            x_generator = self.generator_head(x_generator)
            
            sample_size = mask_positions.int().sum().item()
            flattened_mask_positions = mask_positions.view(-1)

            generator_target = x.clone()
            generator_target[~mask_positions] = -100 # ignore positions
            generator_loss = F.cross_entropy(
                x_generator.view(x.size(0) * x.size(1), -1),
                generator_target.view(-1), 
                reduction='mean',
                ignore_index=-100
            )

            generator_accuracy = (
                torch.max(
                    x_generator.view(
                        x.size(0) * x.size(1), -1), dim=-1
                    )[1][flattened_mask_positions] == \
                x.view(-1)[flattened_mask_positions]
            ).float().mean()

            x_full_generator_result = x.clone()
            x_full_generator_result[mask_positions] = torch.max(x_generator.detach(), dim=-1)[1][mask_positions]

            adjusted_mask_positions = mask_positions.clone()
            adjusted_mask_positions[x_full_generator_result == x] = False
            adjusted_mask_positions = adjusted_mask_positions.float().detach()
            x_full_generator_result = x_full_generator_result.detach()

            # discriminator
            x_discriminator = self.discriminator_lm(x_full_generator_result, input_lengths=x_lengths)
            x_discriminator = self.discriminator_head(x_discriminator).squeeze(-1)

            discriminator_sample_size = length_mask.int().sum().item()
            discriminator_loss = F.binary_cross_entropy_with_logits(
                x_discriminator, adjusted_mask_positions, 
                reduction=('none' if discriminator_sample_size > 0 else 'mean')
            )
            if discriminator_sample_size > 0:
                discriminator_loss = discriminator_loss[length_mask].sum() / discriminator_sample_size

            discriminator_accuracy = \
                (
                    (x_discriminator > .5).float()[mask_positions] == \
                    adjusted_mask_positions[mask_positions]
                ).float().mean()
            loss = generator_loss + self.discriminator_loss_delta * discriminator_loss

            tensorboard_logs = {
                'train_loss': loss,
                'generator_loss': generator_loss,
                'generator_accuracy': generator_accuracy,
                'discriminator_loss': discriminator_loss,
                'discriminator_accuracy': discriminator_accuracy
            }
            return {'loss': loss, 'log': tensorboard_logs}

        def validation_step(self, batch, batch_idx):
            x, x_lengths = batch
            x_lengths = x_lengths.squeeze(1)

            mask_positions, length_mask = self._generate_mask(x, x_lengths)

            x_generator = x.clone()
            x_generator[mask_positions] = self.generator_lm.vocab_size - 1

            x_generator = self.generator_lm(x_generator, input_lengths=x_lengths)
            x_generator = self.generator_head(x_generator)

            sample_size = mask_positions.int().sum().item()
            flattened_mask_positions = mask_positions.view(-1)

            generator_target = x.clone()
            generator_target[~mask_positions] = -100 # ignore positions
            generator_loss = F.cross_entropy(
                x_generator.view(x.size(0) * x.size(1), -1),
                generator_target.view(-1), 
                reduction='mean',
                ignore_index=-100
            )

            generator_accuracy = (
                torch.max(
                    x_generator.view(
                        x.size(0) * x.size(1), -1), dim=-1
                    )[1][flattened_mask_positions] == \
                x.view(-1)[flattened_mask_positions]
            ).float().mean()

            x_full_generator_result = x.clone()
            x_full_generator_result[mask_positions] = torch.max(x_generator.detach(), dim=-1)[1][mask_positions]

            adjusted_mask_positions = mask_positions.clone()
            adjusted_mask_positions[x_full_generator_result == x] = False
            adjusted_mask_positions = adjusted_mask_positions.float().detach()
            x_full_generator_result = x_full_generator_result.detach()

            # discriminator
            x_discriminator = self.discriminator_lm(x_full_generator_result, input_lengths=x_lengths)
            x_discriminator = self.discriminator_head(x_discriminator).squeeze(-1)

            discriminator_sample_size = length_mask.int().sum().item()
            discriminator_loss = F.binary_cross_entropy_with_logits(
                x_discriminator, adjusted_mask_positions, 
                reduction=('none' if discriminator_sample_size > 0 else 'mean')
            )
            if discriminator_sample_size > 0:
                discriminator_loss = discriminator_loss[length_mask].sum() / discriminator_sample_size

            discriminator_accuracy = \
                (
                    (x_discriminator > .5).float()[mask_positions] == \
                    adjusted_mask_positions[mask_positions]
                ).float().mean()

            loss = generator_loss + self.discriminator_loss_delta * discriminator_loss

            result = {
                'batch_first_item': (x[0], x_lengths[0]),
                'first_item_mask': mask_positions[0],
                'first_generator_result': x_full_generator_result[0],
                'first_discriminator_result': x_discriminator[0],
                'val_loss': loss,
                'generator_val_loss': generator_loss,
                'generator_val_accuracy': generator_accuracy,
                'discriminator_val_loss': discriminator_loss,
                'discriminator_val_accuracy': discriminator_accuracy
            }
            return result
        

        def validation_end(self, outputs):
            result = {}
            ignore_keys = [
                'batch_first_item',
                'first_item_mask',
                'first_generator_result',
                'first_discriminator_result'
            ]
            if len(outputs) > 0:
                for key in outputs[0].keys():
                    if key not in ignore_keys:
                        result[key] = torch.stack([x[key] for x in outputs]).mean()
            
            # Sanity check
            MASK_TOKEN = self.generator_lm.vocab_size - 1
            if tokenizer is not None:
                for ix in range(min(5, len(outputs))):
                    x_tokens, x_length = outputs[ix]['batch_first_item']
                    x_tokens = x_tokens[:x_length].cpu().numpy().tolist()
                    item_mask = outputs[ix]['first_item_mask'][:x_length].cpu().numpy().tolist()
                    generator_preds = outputs[ix]['first_generator_result'][:x_length].cpu().numpy().tolist()
                    discriminator_preds = (outputs[ix]['first_discriminator_result'][:x_length] > 0.5).long().cpu().numpy().tolist()

                    original_tokens = [
                        tokenizer.id_to_token(token) for token in x_tokens
                    ]
                    pred_tokens = [
                        tokenizer.id_to_token(token) if item_mask[ix] == 1 else ' ' for 
                            ix, token in enumerate(generator_preds)
                    ]
                    masked_tokens = [
                        '<MASK>' if mask == 1 else ' ' for mask in item_mask
                    ]

                    discriminator_tokens = []

                    for ix in range(len(original_tokens)):
                        orig_token_len = len(original_tokens[ix])
                        pred_token_len = len(pred_tokens[ix])
                        mask_token_len = len(masked_tokens[ix])

                        max_len = max(orig_token_len, pred_token_len, mask_token_len)

                        original_tokens[ix] = original_tokens[ix] + ' ' * (max_len - orig_token_len)
                        pred_tokens[ix] = pred_tokens[ix] + ' ' * (max_len - pred_token_len)
                        masked_tokens[ix] = masked_tokens[ix] + ' ' * (max_len - mask_token_len)
                        discriminator_tokens.append(str(discriminator_preds[ix]) + ' ' * (max_len - 1))

                    original_tokens      = ['Original:     '] + original_tokens
                    masked_tokens        = ['Masked:       '] + masked_tokens
                    pred_tokens          = ['Generator:    '] + pred_tokens
                    discriminator_tokens = ['Discriminator:'] + discriminator_tokens

                    # original_str = tokenizer.decode(x_tokens, skip_special_tokens=False)
                    original_str = ' '.join(original_tokens)
                    masked_str = ' '.join(masked_tokens)
                    pred_str = ' '.join(pred_tokens)
                    disc_str = ' '.join(discriminator_tokens)
                    # pred_str = tokenizer.decode(generator_preds, skip_special_tokens=False)

                    import shutil
                    terminal_size = shutil.get_terminal_size((80, 20))

                    print('---')
                    print(original_str[:terminal_size.columns])
                    print(masked_str[:terminal_size.columns])
                    print(pred_str[:terminal_size.columns])
                    print(disc_str[:terminal_size.columns])

                print('---')
            return result

        def configure_optimizers(self):
            from lamb_optimizer import Lamb
            
            optimizer = Lamb(self.parameters(), lr=1e-3)

            return optimizer
            # num_warmup_steps = 5000
            # num_training_steps = -1
            # weight_decay=0.01
            # from torch.optim.lr_scheduler import LambdaLR
            # def lr_lambda(current_step):
            #     if current_step < num_warmup_steps:
            #         return float(current_step) / float(max(1, num_warmup_steps))
            #     return 1

            # optimizer = Lamb(optimizer_grouped_parameters, lr=1e-3)

            # scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
            
            # return [optimizer], [scheduler]

        @pl.data_loader
        def train_dataloader(self):
            return DataLoader(
                HDF5Dataset(train_dataset_path, 
                    override_length=500000), 
                batch_size=BATCH_SIZE, 
                num_workers=NUM_WORKERS, 
                pin_memory=True
            )

        @pl.data_loader
        def test_dataloader(self):
            return DataLoader(
                HDF5Dataset(test_dataset_path), 
                batch_size=BATCH_SIZE, 
                num_workers=NUM_WORKERS, 
                pin_memory=True
            )

        @pl.data_loader
        def val_dataloader(self):
            return DataLoader(HDF5Dataset(test_dataset_path), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint

    MODEL_CONFIG = HParams({
        'generator_lm': {
            'vocab_size': 12008,
            'embedding_size': 128,
            'embedding_factor_size': 300,
            'recurrent_dropout': .25,
            'hidden_size': 512,
            'n_layers': 3
        },
        'generator_head': {
            'encoder_hidden_size': 512 * 2,
            'vocab_size': 12008,
            'embedding_size': 128,
            'embedding_factor_size': 300
        },
        'discriminator_lm': {
            'vocab_size': 12008,
            'embedding_size': 128,
            'embedding_factor_size': 300,
            'recurrent_dropout': .25,
            'hidden_size': 1152,
            'n_layers': 3
        },
        'discriminator_head': {
            'encoder_hidden_size': 1152 * 2,
            'hidden_size': 512,
            'num_classes': 1
        },
        'discriminator_loss_delta': 25,
        'tie_encoder': False,
        'tie_decoder': True
    })

    model = LMAdversarialModel(MODEL_CONFIG)

    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        save_best_only=False,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )
    trainer = Trainer(
        gpus=1, 
        use_amp=True, 
        row_log_interval=10,
        early_stop_callback=False,
        checkpoint_callback=checkpoint_callback,
        val_percent_check=0.2
    )
    trainer.fit(model)