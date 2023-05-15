from torch import nn
import torch.nn.functional as F
import torch

class BiLSTM(nn.Module):
    
    def __init__(self, label_num, vocab_num, embed_size, pad_idx, **kwargs):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_num, embed_size, padding_idx = pad_idx, max_norm = 1)
        if kwargs["structure"] == "dropout":
            self.embedding_dropout = nn.Dropout(kwargs["dropout_rate"])
        self.bilstm = nn.LSTM(embed_size, 100, num_layers=1, bidirectional=True, batch_first = True)
        self.fc = nn.Linear(200, label_num)

    def forward(self, batch):
        x = self.embedding(batch)
        if hasattr(self, "embedding_dropout"):
            x = self.embedding_dropout(x)
        bilstm_out, _ = self.bilstm(x)
        bilstm_out = F.relu(bilstm_out)
        bilstm_out = bilstm_out.transpose(1,2)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        y = self.fc(bilstm_out)
        return y
    
    @classmethod
    def from_pretrained(cls, label_num, embedding, pad_idx, **kwargs):
        vocab_num, embedding_size = embedding.size()
        model = cls(label_num, vocab_num, embedding_size, pad_idx, **kwargs)
        model.embedding.weight.data = embedding
        return model


class GRU(nn.Module):
    
    def __init__(self, label_num, vocab_num, embed_size, pad_idx, **kwargs):
        super(GRU, self).__init__()
        self.embedding = nn.Embedding(vocab_num, embed_size, padding_idx = pad_idx, max_norm = 1)
        if kwargs["structure"] == "dropout":
            self.embedding_dropout = nn.Dropout(kwargs["dropout_rate"])
        self.gru = nn.GRU(embed_size, 100, num_layers=1, batch_first = True)
        self.fc = nn.Linear(100, label_num)

    def forward(self, batch):
        x = self.embedding(batch)
        if hasattr(self, "embedding_dropout"):
            x = self.embedding_dropout(x)
        gru_out, _ = self.gru(x)
        gru_out = F.relu(gru_out)
        gru_out = gru_out.transpose(1,2)
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)
        y = self.fc(gru_out)
        return y
        
    @classmethod
    def from_pretrained(cls, label_num, embedding, pad_idx, **kwargs):
        vocab_num, embedding_size = embedding.size()
        model = cls(label_num, vocab_num, embedding_size, pad_idx, **kwargs)
        model.embedding.weight.data = embedding
        return model

class CNN(nn.Module):

    def __init__(self, label_num, vocab_num, embed_size, pad_idx, **kwargs):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_num, embed_size, padding_idx = pad_idx, max_norm = 1)
        if kwargs["structure"] == "dropout":
            self.embedding_dropout = nn.Dropout(kwargs["dropout_rate"])
        
        self.conv1 = nn.Conv1d(embed_size, 100, 3, padding=1, stride=1)
        self.conv2 = nn.Conv1d(embed_size, 100, 4, padding=2, stride=1)
        self.conv3 = nn.Conv1d(embed_size, 100, 5, padding=2, stride=1)
        self.fc1 = nn.Linear(300, label_num)

    def forward(self, batch):
        x = self.embedding(batch).transpose(1,2)
        if hasattr(self, "embedding_dropout"):
            x = self.embedding_dropout(x)
            
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(x))
        c3 = F.relu(self.conv3(x))

        z1 = F.max_pool1d(c1, c1.size(2)).view(batch.size(0), -1)
        z2 = F.max_pool1d(c2, c2.size(2)).view(batch.size(0), -1)
        z3 = F.max_pool1d(c3, c3.size(2)).view(batch.size(0), -1)

        z = torch.cat((z1, z2, z3), dim=1)
        y = self.fc1(z)
        return y
    
    @classmethod
    def from_pretrained(cls, label_num, embedding, pad_idx, **kwargs):
        vocab_num, embedding_size = embedding.size()
        model = cls(label_num, vocab_num, embedding_size, pad_idx, **kwargs)
        model.embedding.weight.data = embedding
        return model