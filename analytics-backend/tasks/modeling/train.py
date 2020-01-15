import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random

from torch.utils.data import DataLoader, Dataset
from os import path, getcwd

import pytorch_lightning as pl

import h5py

train_dataset_path = path.join(getcwd(), 'tasks/modeling/data/data_train.h5')
test_dataset_path = path.join(getcwd(), 'tasks/modeling/data/data_test.h5')
BATCH_SIZE = 80
NUM_WORKERS = 7

class HDF5Dataset(Dataset):

    def __init__(self, h5_file_path, dataset_key="tokens", dataset_length_key="lengths", override_length=None):
        super(HDF5Dataset, self).__init__()

        self.h5_file_path = h5_file_path
        self.h5_file = None
        self.dataset_key = dataset_key
        self.dataset_length_key = dataset_length_key
        self.override_length = override_length

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
        return (tokens, length)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    # torch.autograd.set_detect_anomaly(True)

    MAX_SEQUENCE_LENGTH = 128

    from model_lstm import LSTM_LM, LMClassifierHead, LMGeneratorHead

    class LMAdversarialModel(pl.LightningModule):

        def __init__(self):
            super(LMAdversarialModel, self).__init__()

            self.generator_lm = LSTM_LM({
                'vocab_size': 12008,
                'embedding_size': 128,
                'embedding_factor_size': 300,
                'hidden_size': 800,
                'n_layers': 2
            })
            self.discriminator_lm = LSTM_LM({
                'vocab_size': 12008,
                'embedding_size': 128,
                'embedding_factor_size': 300,
                'hidden_size': 1152,
                'n_layers': 3
            })

            self.generator_lm.embedding.weight = self.discriminator_lm.embedding.weight
            self.discriminator_lm.embedding_linear.weight = self.discriminator_lm.embedding_linear.weight

            self.generator_head = LMGeneratorHead({
                'encoder_hidden_size': 800,
                'vocab_size': 12008,
                'embedding_size': 128,
                'embedding_factor_size': 300
            })
            self.discriminator_head = LMClassifierHead({
                'encoder_hidden_size': 1152,
                'hidden_size': 512,
                'num_classes': 1
            })
            self.discriminator_loss_delta = 50.

            self.generator_head.decoder.weight = self.generator_lm.embedding.weight

        def forward(self, tokens):
            return self.discriminator_lm(tokens)

        def training_step(self, batch, batch_idx):
            x, x_lengths = batch
            x_lengths = x_lengths.squeeze(1)

            # indices to mask out
            mask_probs = torch.rand(x.size(0), x.size(1)).cuda()
            mask_positions = (mask_probs <= .15).cuda().detach()
            mask_targets = torch.randint(self.generator_lm.vocab_size - 2, (x.size(0), x.size(1))).cuda()
            
            mask_targets[~mask_positions] = x[~mask_positions]

            x_generator = x
            x_generator[mask_positions] = self.generator_lm.vocab_size - 1

            # generator
            x_generator = self.generator_lm(x_generator, input_lengths=x_lengths)

            sample_size = mask_positions.int().sum().item()

            # if sample_size == 0:
            # loss on the whole batch
            x_generator = self.generator_head(x_generator)
            generator_loss = F.cross_entropy(
                x_generator.view(x.size(0) * x.size(1), -1),
                mask_targets.view(-1), 
                reduction='mean'
            )
            generator_accuracy = (torch.max(x_generator.view(x.size(0) * x.size(1), -1), dim=-1)[1] == mask_targets.view(-1))\
                .float().mean()

            x_full_generator_result = torch.max(x_generator.detach(), dim=-1)[1]
            # else:
            #     # loss on only masked tokens
            #     x_generator = x_generator[mask_positions,:]
            #     x_generator = self.generator_head(x_generator)
            #     generator_loss = F.cross_entropy(
            #         x_generator,
            #         mask_targets[mask_positions].view(-1), 
            #         reduction='sum'
            #     ) / sample_size

            #     x_full_generator_result = x.clone()
            #     x_full_generator_result[mask_positions] = torch.max(x_generator.detach(), dim=-1)[1]
            
            mask_positions[x_full_generator_result == mask_targets] = False
            mask_positions = mask_positions.float().detach()
            x_full_generator_result = x_full_generator_result.detach()

            # discriminator
            x_discriminator = self.discriminator_lm(x_full_generator_result, input_lengths=x_lengths)
            x_discriminator = self.discriminator_head(x_discriminator).squeeze(-1)

            discriminator_loss = F.binary_cross_entropy_with_logits(
                x_discriminator, mask_positions, 
                reduction='mean'
            )
            discriminator_accuracy = ((x_discriminator > .5).float() == mask_positions)\
                .float().mean()

            loss = generator_loss + discriminator_loss

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

            mask_probs = torch.rand(x.size(0), x.size(1)).cuda()
            mask_positions = (mask_probs <= .15).cuda().detach()
            mask_targets = torch.randint(self.generator_lm.vocab_size - 2, (x.size(0), x.size(1))).cuda()
            
            mask_targets[~mask_positions] = x[~mask_positions]

            x_generator = x
            x_generator[mask_positions] = self.generator_lm.vocab_size - 1

            x_generator = self.generator_lm(x_generator, input_lengths=x_lengths)

            sample_size = mask_positions.int().sum().item()

            x_generator = self.generator_head(x_generator)
            generator_loss = F.cross_entropy(
                x_generator.view(x.size(0) * x.size(1), -1),
                mask_targets.view(-1), 
                reduction='mean'
            )
            generator_accuracy = (torch.max(x_generator.view(x.size(0) * x.size(1), -1), dim=-1)[1] == mask_targets.view(-1))\
                .float().mean()

            x_full_generator_result = torch.max(x_generator.detach(), dim=-1)[1]

            mask_positions[x_full_generator_result == mask_targets] = False
            mask_positions = mask_positions.float().detach()
            x_full_generator_result = x_full_generator_result.detach()

            # discriminator
            x_discriminator = self.discriminator_lm(x_full_generator_result, input_lengths=x_lengths)
            x_discriminator = self.discriminator_head(x_discriminator).squeeze(-1)

            discriminator_loss = F.binary_cross_entropy_with_logits(
                x_discriminator, mask_positions, 
                reduction='mean'
            )
            discriminator_accuracy = ((x_discriminator > .5).float() == mask_positions)\
                .float().mean()

            loss = generator_loss + discriminator_loss

            result = {
                'val_loss': loss,
                'generator_val_loss': generator_loss,
                'generator_val_accuracy': generator_accuracy,
                'discriminator_val_loss': discriminator_loss,
                'discriminator_val_accuracy': discriminator_accuracy
            }
            return result
        

        def validation_end(self, outputs):
            result = {}
            if len(outputs) > 0:
                for key in outputs[0].keys():
                    result[key] = torch.stack([x[key] for x in outputs]).mean()
            
            return result

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters())

        @pl.data_loader
        def train_dataloader(self):
            return DataLoader(HDF5Dataset(train_dataset_path, override_length=100000), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

        @pl.data_loader
        def test_dataloader(self):
            return DataLoader(HDF5Dataset(test_dataset_path), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

        @pl.data_loader
        def val_dataloader(self):
            return DataLoader(HDF5Dataset(test_dataset_path), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint

    model = LMAdversarialModel()
    trainer = Trainer(
        gpus=1, 
        use_amp=True, 
        row_log_interval=10,
        early_stop_callback=False,
        checkpoint_callback=True
    )
    trainer.fit(model)