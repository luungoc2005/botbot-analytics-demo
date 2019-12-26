import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

from config import DATA_DIR, CACHE_DIR
from common import cache, utils, ignore_lists, dialogflow
from tasks.common import return_response

from os import listdir, path, makedirs

DEFAULT_CONFIG = {
    'num_classes': 5,
    'embedding_size': 300,
    'context_size': 20,
    'hidden_size': 50
}

class DfTrainingDataset(Dataset):

    def __init__(self, file_path):
        file_hash = cache.get_file_hash(file_path)

        if not path.exists(CACHE_DIR) or not path.isdir(CACHE_DIR):
            makedirs(CACHE_DIR)
        
        cache_path = path.join(CACHE_DIR, f'model_{file_hash}.bin')

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

        le = LabelEncoder()
        X_train = utils.get_sentence_vectors(raw_exampes_tokens)
        mlb = None

        if has_contexts:
            print('Featurizing contexts')
            mlb = MultiLabelBinarizer()
            X_contexts = mlb.fit_transform(raw_contexts)

        y_train = le.fit_transform(raw_labels)

        self.X_train = X_train
        self.X_contexts = X_contexts
        self.y_train = y_train

    def __len__(self):
        return len(self.X_train)

    def __getitem__(self, idx):
        return X_train[idx], X_contexts[idx], y_train[idx]


class LSTMTextClassifier(pl.LightningModule):

    def __init__(self, config={}, train_dataset=None):
        super(LSTMTextClassifier, self).__init__()

        self.config = DEFAULT_CONFIG.update(config)
        self.num_classes = self.config['num_classes']
        self.embedding_size = self.config['embedding_size']
        self.context_size = self.config['context_size']
        self.hidden_size = self.config['hidden_size']
        self.train_dataset = train_dataset

        self.rnn = nn.LSTM(
            self.embedding_size + self.context_size, self.hidden_size,
            bidirectional=True,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2 + self.context_size, self.hidden_size),
            nn.Linear(self.hidden_size, self.num_classes)
        )

    def forward(self, examples_input, context_input):
        x = self.rnn(examples_input)[0]
        
        x = x[:,-1] # last token state
        x = torch.concat((x, context_input), axis=-1)

        x = self.classifier(x)

        return x

    def training_step(self, batch, batch_idx):
        examples_input, context_input, y = batch
        y_hat = self.forward(examples_input, context_input)
        loss = F.cross_entropy(y_hat, y)
        
        return { 'loss': loss }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    @pl.data_loader
    def train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError('Model was initialized without any training dataset')
        return DataLoader(self.train_dataset, batch_size=32)
        