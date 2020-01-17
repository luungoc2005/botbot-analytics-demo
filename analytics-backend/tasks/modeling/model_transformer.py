# https://pytorch.org/tutorials/beginner/transformer_tutorial.html?highlight=transformer


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm

DEFAULT_TRANSFORMER_CONFIG = {
    'vocab_size': 12000,
    'embedding_size': 128,
    'embedding_factor_size': 768,
    'num_attention_heads': 4,
    'max_sequence_length': 128,
    'dim_feedforward': 512,
    'num_layers': 12,
    'dropout': .1
}


class TransformerLM(nn.Module):

    def __init__(self, config):
        super(TransformerLM, self).__init__()

        self.config = DEFAULT_TRANSFORMER_CONFIG
        self.config.update(config)

        self.vocab_size = self.config['vocab_size']
        self.embedding_size = self.config['embedding_size']
        self.embedding_factor_size = self.config['embedding_factor_size']
        self.num_attention_heads = self.config['num_attention_heads']
        self.max_sequence_length = self.config['max_sequence_length']
        self.dim_feedforward = self.config['dim_feedforward']
        self.num_layers = self.config['num_layers']
        self.dropout = self.config['dropout']

        self.pos_encoder = PositionalEncoding(self.embedding_size, 0., 
            max_len=self.max_sequence_length
        )

        encoder_layers = TransformerEncoderLayer(
            self.vocab_size,
            self.num_attention_heads, 
            self.dim_feedforward,
            self.dropout
        )
        encoder_norm = LayerNorm(self.embedding_factor_size)
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, 
            self.num_layers,
            encoder_norm
        )

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embedding_linear = nn.Linear(self.embedding_size, self.embedding_factor_size)


    def forward(self, tokens, input_lengths=None):
        x = self.embedding(tokens)
        x = self.embedding_linear(x)

        x = self.encoder(x) * math.sqrt(self.embedding_factor_size)
        x = self.pos_encoder(x)

        if input_lengths is None:
            x = self.transformer_encoder(x)
        else:
            mask = torch.arange(self.max_sequence_length).unsqueeze(0) < input_lengths.unsqueeze(1)
            mask = torch.where(mask, float('-inf'), 0.)
            x = self.transformer_encoder(x, mask)
        
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
