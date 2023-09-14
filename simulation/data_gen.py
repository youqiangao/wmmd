import numpy as np
import torch 
from torch.utils.data import Dataset
from scipy.stats import betabinom

class DataGenerator(object):
    def __init__(self, vocab_num, word_size, dist0_alpha, dist0_beta, dist1_alpha, dist1_beta):
        super(DataGenerator, self).__init__()
        self.vocab_num = vocab_num 
        self.word_size = word_size
        self.dist0_alpha = dist0_alpha
        self.dist0_beta = dist0_beta
        self.dist1_alpha = dist1_alpha
        self.dist1_beta = dist1_beta
        
    def generate(self, sample_size):
        labels = np.random.choice(np.arange(2), size = (sample_size,))
        n = self.vocab_num
        rv0 = betabinom(n - 1 , self.dist0_alpha, self.dist0_beta) 
        rv1 = betabinom(n - 1 , self.dist1_alpha, self.dist1_beta)
        input_ids = []
        for idx in labels:
            if idx == 0:
                input_ids.append(rv0.rvs(self.word_size))
            else:
                input_ids.append(rv1.rvs(self.word_size))
        labels = np.int32(labels)
        input_ids = np.array(input_ids)
        data = {'input_ids': input_ids, 'label': labels}
        return data
        
        
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, idx):
        input_ids, label = self.data['input_ids'][idx], self.data['label'][idx]
        input_ids = torch.tensor(input_ids, dtype = torch.long)
        label = torch.tensor(label, dtype = torch.float) # float is for the calculation of BCELoss
        return {'input_ids': input_ids, 'label': label}
    

    
    