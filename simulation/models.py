import torch
from torch import nn

class LogisticRegression(torch.nn.Module):
    def __init__(self, vocab_num, embedding_size, word_size, **kwargs):
        super(LogisticRegression, self).__init__()
        self.embedding = nn.Embedding(vocab_num, embedding_size, max_norm = 1) 
        self.linear = torch.nn.Linear(embedding_size * word_size , 1)
        if kwargs["structure"] == "dropout":
            self.dropout = nn.Dropout(kwargs["dropout_rate"])
            
        
    def forward(self, vocab_id, **kwargs):
        x = self.embedding(vocab_id)
        x = torch.flatten(x, start_dim = 1)
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        outputs = torch.sigmoid(self.linear(x)).squeeze(1)
        return outputs