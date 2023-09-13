import torch 
import numpy as np


class wMMD(object):
    """
    Compute Maximum Mean Discrepancy using mini-batch data.
    """
    
    def __init__(self, embedding, stopping_idx, device):
        self.embedding = embedding
        self.vocab_num = embedding.size()[0]
        self.stopping_idx = stopping_idx
        self.device = device

    def compute(self, ids, labels):
        mmd2_total = torch.tensor(0)
        unique_labels = torch.unique(labels)
        K = len(unique_labels) # the number of unique labels in mini-batch data
        if K ==  1:
            return torch.tensor(0)
        for i, label0 in enumerate(unique_labels):
            for label1 in unique_labels[i+1:K]:
                ids_tmp = ids[torch.logical_or(labels == label0 , labels == label1)]
                labels_tmp = labels[torch.logical_or(labels == label0 , labels == label1)]
                mmd2 = self.pair_compute(ids_tmp, labels_tmp)
                mmd2_total = mmd2_total + mmd2
        return mmd2_total * 2 / (K * (K - 1))


    def pair_compute(self, ids, labels):
     
        dist_mat = DIST_MAT(self.embedding).compute(ids)
        
        word_pair_count00, word_pair_count01, word_pair_count11 = WORD_PAIR_COUNT(self.stopping_idx, self.device).compute(ids, labels)
        word_pair_count = word_pair_count00 + 2 * word_pair_count01 + word_pair_count11

        sigma2 = self._median(dist_mat, word_pair_count) 
        gamma = 1/(2 * sigma2)
        K_XX = torch.exp(-gamma * dist_mat)
        if torch.sum(word_pair_count00) == 0 or torch.sum(word_pair_count11) == 0:
            return torch.tensor(0)
        else:
            mmd2 = torch.sum(K_XX * word_pair_count00)/torch.sum(word_pair_count00) \
                    + torch.sum(K_XX * word_pair_count11)/torch.sum(word_pair_count11) \
                    - 2 * torch.sum(K_XX * word_pair_count01)/torch.sum(word_pair_count01)
        return mmd2
    
    def _median(self, values, counts):
        """
        find the median based on values and counts
        """
        values = torch.flatten(values.detach())
        counts = torch.flatten(counts)
        return torch.median(torch.repeat_interleave(values, counts)) + 1e-8


class WORD_PAIR_COUNT():
    """
    Count all word pairs in the dataset.
    """
    def __init__(self, stopping_idx, device):
        self.stopping_idx = stopping_idx
        self.device = device
    
    def compute(self, ids, labels):
        ids = ids.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        ids_unique = np.unique(ids)
        labels_unique = np.unique(labels)
        vocab = {idx:i for i, idx in enumerate(ids_unique) } # vocabulary set mapping a vocabulary to an index
        
        count_mat = np.array([self.sentence_count(id, vocab) for id in ids])
        count_mat0 = count_mat[labels == labels_unique[0]]
        count_mat1 = count_mat[labels == labels_unique[1]]
        
        colsum = lambda mat: np.sum(mat, 0)
        count_fun = lambda mat: np.outer(colsum(mat), colsum(mat)) - np.dot(mat.transpose((1,0)), mat)
        t = lambda mat: self.tensor(mat)
        pair_count00 = t(count_fun(count_mat0))
        pair_count11 = t(count_fun(count_mat1))
        pair_count01 = t(np.outer(colsum(count_mat0), colsum(count_mat1)))
        
        return (pair_count00, pair_count01, pair_count11)
    
    def sentence_count(self, sentence, vocab):
        words, counts = np.unique(sentence, return_counts = True)
        tmp1, tmp2 = [], []
        # exclude stopping words
        for word, count in zip(words, counts):
            if word not in self.stopping_idx:
                tmp1.append(word)
                tmp2.append(count)
        words, counts = tmp1, tmp2 
        indice = [vocab[word] for i, word in enumerate(words)]
        count_array = np.zeros((len(vocab), ))
        count_array[indice] = counts
        return count_array
    
    def tensor(self, array):
        """
        transform the np.array to torch.tensor 
        """
        tensor = torch.tensor(array).to(torch.int64).to(self.device)
        return tensor
    
class DIST_MAT():
    """
    Compute all distances between all possible word pairs in the dictionary.
    """
    def __init__(self, embedding):
        self.embedding = embedding
        
    def compute(self, ids):
        ids = torch.unique(ids, sorted=True)
        X = self.embedding[ids]
        XX = torch.matmul(X, X.transpose(1,0))
        X_sqnorms = torch.diag(XX)
        r = lambda x: torch.unsqueeze(x, 0)
        c = lambda x: torch.unsqueeze(x, 1)
        D_XX = -2 * XX + c(X_sqnorms) + r(X_sqnorms)
        return D_XX

class L1(object):
    def __init__(self, vocab_embedding):
        self.vocab_embedding = vocab_embedding
        
    def compute(self):
        return torch.mean(torch.abs(self.vocab_embedding))