import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score

from config import DATA_DIR, CACHE_DIR
from common import cache, utils, ignore_lists, dialogflow
from tasks.common import return_response

from os import listdir, path, makedirs

import json

class DfTrainingDataset(Dataset):

    def __init__(self, file_path):

        # TODO: support priority

        if file_path.lower()[-5:] == '.json':
            with open(file_path, 'r') as input_file:
                training_file = json.load(input_file)
        else:
            training_file = dialogflow.load_dialogflow_archive(file_path)

        raw_examples = []
        raw_labels = []
        raw_contexts = []
        # raw_priorities = []
        examples_counts = {}
        has_contexts = False

        for intent_idx, intent in enumerate(training_file):
            intent_name = intent['name']
            intent_priority = intent.get('priority', 500000)

            if intent_priority > 0:
                for usersay in intent['usersays']:
                    raw_labels.append(intent_name)
                    raw_examples.append(usersay.strip())
                    contexts = intent.get('contexts', []) # this is optional
                    # raw_priorities.append([intent_priority])
                    if len(contexts) > 0:
                        has_contexts = True
                    raw_contexts.append(contexts)

            examples_counts[intent_name] = len(intent['usersays'])

        raw_exampes_tokens = utils.tokenize_text_list(raw_examples)

        self.raw_examples = raw_examples
        self.raw_labels = raw_labels
        self.raw_contexts = raw_contexts
        self.examples_counts = examples_counts
        self.has_contexts = has_contexts
        
        le = LabelEncoder()
        X_train = utils.get_sentence_vectors_full(raw_exampes_tokens)
        X_contexts = None
        mlb = None

        if has_contexts:
            print('Featurizing contexts')
            mlb = MultiLabelBinarizer()
            X_contexts = mlb.fit_transform(raw_contexts)

        y_train = le.fit_transform(raw_labels)

        self.le = le
        self.mlb = mlb

        self.X_train = torch.from_numpy(X_train).float()
        self.X_contexts = torch.from_numpy(X_contexts).float() \
            if X_contexts is not None else None
        self.y_train = torch.from_numpy(y_train).long()

        self.embedding_size = self.X_train.size(-1)
        self.context_size = self.X_contexts.size(-1) \
            if X_contexts is not None else 0
        self.num_classes = len(le.classes_)
        self.classes_ = le.classes_

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        if self.has_contexts:
            return self.X_train[idx], self.X_contexts[idx], self.y_train[idx]
        else:
            return self.X_train[idx], self.y_train[idx]


DEFAULT_LSTM_CONFIG = {
    'num_classes': 5,
    'embedding_size': 300,
    'embedding_factor_size': 128,
    'context_size': 20,
    'rnn_size': 30,
    'hidden_size': 50
}

class LSTMTextClassifier(pl.LightningModule):

    def __init__(self, config={}, train_dataset=None):
        super(LSTMTextClassifier, self).__init__()

        self.config = DEFAULT_LSTM_CONFIG
        self.config.update(config)

        self.num_classes = self.config['num_classes']
        self.embedding_size = self.config['embedding_size']
        self.context_size = self.config['context_size']
        self.rnn_size = self.config['rnn_size']
        self.hidden_size = self.config['hidden_size']
        self.embedding_factor_size = self.config['embedding_factor_size']
        self.train_dataset = train_dataset
        self.has_contexts = train_dataset.has_contexts

        self.emb_factor_layer = nn.Linear(
            self.embedding_size,
            self.embedding_factor_size
        )

        self.rnn = nn.LSTM(
            self.embedding_factor_size, self.rnn_size,
            bidirectional=True,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.rnn_size * 2 + self.context_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.num_classes)
        )
        # self.classifier = nn.Linear(self.hidden_size * 2 + self.context_size, self.num_classes)

    def forward(self, examples_input, context_input):
        x = self.emb_factor_layer(examples_input)
        x = self.rnn(x)[0]
        
        # x = x[:,-1] # last token state
        x = torch.max(x, dim=1)[0] # max pooling
        if (self.has_contexts):
            x = torch.cat((x, context_input), dim=-1)

        x = self.classifier(x)

        return x

    def training_step(self, batch, batch_idx):
        if self.has_contexts:
            examples_input, context_input, y = batch
        else:
            examples_input, y = batch 
            context_input = None
        y_hat = self.forward(examples_input, context_input)
        loss = F.cross_entropy(y_hat, y)
        
        # accuracy calculation
        y_hat_classes = torch.max(y_hat, dim=1)[0]
        accuracy = (y_hat_classes == y).float().mean().item()

        tensorboard_logs = {'train_loss': loss, 'train_acc': accuracy}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    @pl.data_loader
    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError('Model was initialized without any training dataset')
        return DataLoader(self.train_dataset, batch_size=200)
        

DEFAULT_CNN_CONFIG = {
    'num_classes': 5,
    'embedding_size': 300,
    'embedding_factor_size': 128,
    'context_size': 20,
    'filter_sizes': [3, 4, 5],
    'dropout': .2,
    'n_filters': 100,
    'hidden_size': 50
}

class CNNTextClassifier(pl.LightningModule):

    def __init__(self, config={}, train_dataset=None):
        super(CNNTextClassifier, self).__init__()

        self.config = DEFAULT_CNN_CONFIG
        self.config.update(config)

        self.num_classes = self.config['num_classes']
        self.embedding_size = self.config['embedding_size']
        self.context_size = self.config['context_size']
        self.hidden_size = self.config['hidden_size']
        self.filter_sizes = self.config['filter_sizes']
        self.n_filters = self.config['n_filters']
        self.dropout_val = self.config['dropout']
        self.embedding_factor_size = self.config['embedding_factor_size']
        self.train_dataset = train_dataset
        self.has_contexts = train_dataset.has_contexts
        self.batch_size = 64

        self.emb_factor_layer = nn.Linear(
            self.embedding_size,
            self.embedding_factor_size
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.embedding_factor_size, 
                out_channels=self.n_filters, 
                kernel_size=fs
            )
            for fs in self.filter_sizes
        ])

        self.dropout = nn.Dropout(self.dropout_val)

        self.classifier = nn.Sequential(
            nn.Linear(
                len(self.filter_sizes) * self.n_filters + self.context_size, 
                self.hidden_size
            ),
            nn.Linear(self.hidden_size, self.num_classes)
        )
        # self.classifier = nn.Linear(self.hidden_size * 2 + self.context_size, self.num_classes)

    def forward(self, examples_input, context_input):
        x = self.emb_factor_layer(examples_input)

        x = x.permute(0, 2, 1)
        #embedded = [batch size, emb dim, sent len]

        x = [F.relu(conv(x)) for conv in self.convs]
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        x = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in x]
        #pooled_n = [batch size, n_filters]

        x = self.dropout(torch.cat(x, dim=1))

        if (self.has_contexts):
            x = torch.cat((x, context_input), dim=-1)

        x = self.classifier(x)

        return x

    def training_step(self, batch, batch_idx):
        if self.has_contexts:
            examples_input, context_input, y = batch
        else:
            examples_input, y = batch 
            context_input = None
        y_hat = self.forward(examples_input, context_input)
        loss = F.cross_entropy(y_hat, y)
        
        # accuracy calculation
        y_hat_classes = torch.max(y_hat, dim=1)[0]
        accuracy = (y_hat_classes == y).float().mean().item()

        tensorboard_logs = {'train_loss': loss, 'train_acc': accuracy}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # return torch.optim.SGD(self.parameters())
        opt = torch.optim.Adam(self.parameters())
        # opt = torch.optim.SGD(
        #     self.parameters(), 
        #     lr=0
        #     momentum=0.9
        # )
        sched = optim.lr_scheduler.OneCycleLR(
            opt, 
            max_lr=0.01, 
            steps_per_epoch=int(len(self.train_dataset) // self.batch_size), 
            epochs=20
        )
        # sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     opt,
        #     int(len(self.train_dataset) // self.batch_size) + 1
        # )
        return [opt], [sched]

    @pl.data_loader
    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError('Model was initialized without any training dataset')
        return DataLoader(self.train_dataset, batch_size=self.batch_size)
        