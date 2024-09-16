
import torch
from torch import nn
import torch.nn.functional as F
from regularizers import Regularizer, Dropout
from typing import Optional


class Model(torch.nn.Module):
    def _padded_embedding(self, padding_idx: Optional[int] = None) -> torch.nn.Module:
        if not padding_idx:
            embedding_layer = nn.Embedding(self.vocab_num, self.embedding_size, padding_idx = padding_idx, max_norm = 1)
        else: 
            embedding_layer = nn.Embedding(self.vocab_num, self.embedding_size, max_norm = 1)
        return embedding_layer

class LogisticRegression(Model):
    def __init__(
        self, 
        vocab_num: int, 
        embedding_size: int, 
        word_size: int, 
        num_classes: int, 
        regularizer: Regularizer,
        padding_idx: Optional[int] = None,
    ) -> None:
        super(LogisticRegression, self).__init__()
        self.vocab_num = vocab_num
        self.embedding_size = embedding_size
        self.embedding_layer = self._padded_embedding(padding_idx)
        self.fc = torch.nn.Linear(embedding_size * word_size , num_classes)
        self.regularizer = regularizer
        if isinstance(self.regularizer, Dropout):
            self.dropout = nn.Dropout(self.regularizer.dropout_rate)
        
    def forward(self, vocab_id: torch.Tensor) -> torch.Tensor:
        x = self.embedding_layer(vocab_id)
        x = torch.flatten(x, start_dim = 1)
        if isinstance(self.regularizer, Dropout):
            x = self.dropout(x)
        outputs = self.fc(x)
        return outputs



class MLP(Model):
    def __init__(
        self, 
        vocab_num: int, 
        embedding_size: int,
        word_size: int,
        num_classes: int,
        regularizer: Regularizer,
        padding_idx: Optional[int] = None,
    ) -> None:
        super(MLP, self).__init__()
        self.vocab_num = vocab_num
        self.embedding_size = embedding_size
        self.embedding_layer = self._padded_embedding(padding_idx)
        self.layers = nn.Sequential(
            nn.Linear(embedding_size * word_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
        self.regularizer = regularizer
        if isinstance(self.regularizer, Dropout):
            self.dropout = nn.Dropout(self.regularizer.dropout_rate)

    def forward(self, vocab_id: torch.Tensor) -> torch.Tensor:
        x = self.embedding_layer(vocab_id)
        x = torch.flatten(x, start_dim = 1)
        if isinstance(self.regularizer, Dropout):
            x = self.dropout(x)
        outputs = self.layers(x)
        return outputs


class BiLSTM(Model):
    def __init__(
        self, 
        vocab_num: int, 
        embedding_size: int, 
        num_classes: int, 
        regularizer: Regularizer,
        padding_idx: Optional[int] = None,
    ) -> None:
        super(BiLSTM, self).__init__()
        self.vocab_num = vocab_num
        self.embedding_size = embedding_size
        self.embedding_layer = self._padded_embedding(padding_idx)
        self.bilstm = nn.LSTM(embedding_size, 100, num_layers=1, bidirectional=True, batch_first = True)
        self.fc = nn.Linear(200, num_classes)
        self.regularizer = regularizer
        if isinstance(self.regularizer, Dropout):
            self.dropout = nn.Dropout(self.regularizer.dropout_rate)

    def forward(self, vocab_id: torch.Tensor) -> torch.Tensor:
        x = self.embedding_layer(vocab_id)
        if isinstance(self.regularizer, Dropout):
            x = self.dropout(x)
        bilstm_out, _ = self.bilstm(x)
        bilstm_out = F.relu(bilstm_out)
        bilstm_out = bilstm_out.transpose(1,2)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        outputs = self.fc(bilstm_out)
        return outputs

class GRU(Model):
    def __init__(
        self, 
        vocab_num: int, 
        embedding_size: int, 
        num_classes: int, 
        regularizer: Regularizer,
        padding_idx: Optional[int] = None,
    ) -> None:
        super(GRU, self).__init__()
        self.vocab_num = vocab_num
        self.embedding_size = embedding_size
        self.embedding_layer = self._padded_embedding(padding_idx)
        self.gru = nn.GRU(embedding_size, 100, num_layers=1, batch_first = True)
        self.fc = nn.Linear(100, num_classes)
        self.regularizer = regularizer
        if isinstance(self.regularizer, Dropout):
            self.dropout = nn.Dropout(self.regularizer.dropout_rate)

    def forward(self, vocab_id: torch.Tensor) -> torch.Tensor:
        x = self.embedding_layer(vocab_id)
        if isinstance(self.regularizer, Dropout):
            x = self.dropout(x)
        gru_out, _ = self.gru(x)
        gru_out = F.relu(gru_out)
        gru_out = gru_out.transpose(1,2)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        outputs = self.fc(gru_out)
        return outputs

class CNN(Model):
    def __init__(
        self, 
        vocab_num: int, 
        embedding_size: int, 
        num_classes: int,
        regularizer: Regularizer,
        padding_idx: Optional[int] = None,
    ) -> None:
        super(CNN, self).__init__()
        self.vocab_num = vocab_num
        self.embedding_size = embedding_size
        self.embedding_layer = self._padded_embedding(padding_idx)
        self.conv1 = nn.Conv1d(self.embedding_size, 100, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(self.embedding_size, 100, 4, padding=2, stride=1)
        self.conv3 = nn.Conv1d(self.embedding_size, 100, 5, padding=2, stride=1)
        self.fc = nn.Linear(300, num_classes)
        self.regularizer = regularizer
        if isinstance(self.regularizer, Dropout):
            self.dropout = nn.Dropout(self.regularizer.dropout_rate)

    def forward(self, vocab_id: torch.Tensor) -> torch.Tensor:
        x = self.embedding_layer(vocab_id).transpose(1,2)
        if isinstance(self.regularizer, Dropout):
            x = self.dropout(x)
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(x))
        c3 = F.relu(self.conv3(x))
        z1 = F.max_pool1d(c1, c1.size(2)).view(vocab_id.size(0), -1)
        z2 = F.max_pool1d(c2, c2.size(2)).view(vocab_id.size(0), -1)
        z3 = F.max_pool1d(c3, c3.size(2)).view(vocab_id.size(0), -1)
        z = torch.cat((z1, z2, z3), dim=1)
        outputs = self.fc(z)
        return outputs