# https://pytorch.org/tutorials/beginner/transformer_tutorial.html?highlight=transformer


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm

DEFAULT_TRANSFORMER_CONFIG = {
    'vocab_size': 12008,
    'embedding_size': 128,
    'embedding_factor_size': 768,
    'num_attention_heads': 4,
    'max_sequence_length': 128,
    'dim_feedforward': 512,
    'num_layers': 12,
    'dropout': .1
}

# https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html
class TiedTransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers
    "Tied": ALBERT-like sharing of parameters across layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TiedTransformerEncoder, self).__init__()
        self.layer = encoder_layer
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
            r"""Pass the input through the encoder layers in turn.

            Args:
                src: the sequnce to the encoder (required).
                mask: the mask for the src sequence (optional).
                src_key_padding_mask: the mask for the src keys per batch (optional).

            Shape:
                see the docs in Transformer class.
            """
            output = src

            for i in range(self.num_layers):
                output = self.layer(output, src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask)

            if self.norm:
                output = self.norm(output)

            return output

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
            self.embedding_factor_size,
            self.num_attention_heads, 
            self.dim_feedforward,
            self.dropout
        )
        encoder_norm = LayerNorm(self.embedding_factor_size)
        self.transformer_encoder = TiedTransformerEncoder(
            encoder_layers, 
            self.num_layers,
            encoder_norm
        )
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embedding_linear = nn.Linear(self.embedding_size, self.embedding_factor_size)


    def forward(self, tokens, input_lengths=None):
        x = self.embedding(tokens)

        # since pytorch transformer layers are seq_length first
        x = x.permute(1, 0, 2)

        x = x * math.sqrt(self.embedding_size)
        x = self.pos_encoder(x)

        x = self.embedding_linear(x)

        if input_lengths is None:
            x = self.transformer_encoder(x)
        else:
            mask = torch.arange(self.max_sequence_length).unsqueeze(0).cuda() < input_lengths.unsqueeze(1)

            x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # then swap back
        x = x.permute(1, 0, 2)

        return x


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
