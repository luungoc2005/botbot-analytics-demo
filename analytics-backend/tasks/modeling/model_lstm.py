import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

DEFAULT_LSTM_CONFIG = {
    'vocab_size': 32000,
    'embedding_size': 128,
    'embedding_factor_size': 300,
    'hidden_size': 2048,
    'n_layers': 3
}

DEFAULT_CLASSIFIER_CONFIG = {
    'encoder_hidden_size': 2048,
    'hidden_size': 512,
    'num_classes': 256
}

DEFAULT_GENERATOR_CONFIG = {
    'encoder_hidden_size': 768,
    'vocab_size': 32000,
    'embedding_size': 128,
    'embedding_factor_size': 300
}

class LSTM_LM(nn.Module):

    def __init__(self, config={}):
        super(LSTM_LM, self).__init__()

        self.config = DEFAULT_LSTM_CONFIG
        self.config.update(config)
        
        self.vocab_size = self.config['vocab_size']
        self.embedding_size = self.config['embedding_size']
        self.embedding_factor_size = self.config['embedding_factor_size']
        self.hidden_size = self.config['hidden_size']
        self.n_layers = self.config['n_layers']
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embedding_linear = nn.Linear(self.embedding_size, self.embedding_factor_size)
        self.rnn = nn.LSTM(self.embedding_factor_size, self.hidden_size,
            batch_first=True,
            bidirectional=True, 
            num_layers=self.n_layers
        )
        
    def forward(self, tokens, pool=False):
        x = self.embedding(tokens)
        x = self.embedding_linear(x)

        x = self.rnn(x)[0]

        if pool:
            # x[x == 0] = -1e8
            x = torch.max(x, 1)[0]
            if x.ndimension() == 3:
                x = x.squeeze(1)
                assert x.ndimension() == 2

        return x


class LMClassifierHead(nn.Module):

    def __init__(self, config={}):
        super(LMClassifierHead, self).__init__()
        
        self.config = DEFAULT_CLASSIFIER_CONFIG
        self.config.update(config)

        self.encoder_hidden_size = self.config['encoder_hidden_size']
        self.hidden_size = self.config['hidden_size']
        self.num_classes = self.config['num_classes']

        self.classifier = nn.Sequential(
            nn.Linear(self.encoder_hidden_size * 2, self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class LMGeneratorHead(nn.Module):

    def __init__(self, config={}):
        super(LMGeneratorHead, self).__init__()
        
        self.config = DEFAULT_GENERATOR_CONFIG
        self.config.update(config)

        self.encoder_hidden_size = self.config['encoder_hidden_size']
        self.embedding_size = self.config['embedding_size']
        self.embedding_factor_size = self.config['embedding_factor_size']
        self.vocab_size = self.config['vocab_size']

        self.linear = nn.Linear(self.encoder_hidden_size * 2, self.embedding_factor_size)
        self.embedding_linear = nn.Linear(self.embedding_factor_size, self.embedding_size)
        self.decoder = nn.Linear(self.embedding_size, self.vocab_size)

    def forward(self, x):
        x = self.linear(x)
        x = self.embedding_linear(x)
        return self.decoder(x)