import numpy as np
import torch
from typing import List

class Regularizer():
    def __init__(self) -> None:
        pass

    def assign_embedding(self, vocab_embedding: torch.Tensor) -> None:
        self.vocab_embedding = vocab_embedding

class NoneReg(Regularizer):
    def __call__(self, **kwargs) -> torch.Tensor:
        return torch.tensor(0.0)

class L1(Regularizer):
    def __init__(self, weight: float) -> None:
        self.weight = weight
        
    def __call__(self, **kwargs) -> torch.Tensor:
        return self.weight * torch.mean(torch.abs(self.vocab_embedding))

class Dropout(Regularizer):
    def __init__(self, dropout_rate: float) -> None:
        self.dropout_rate = dropout_rate

    def __call__(self, **kwargs) -> torch.Tensor:
        return torch.tensor(0.0)



class wMMD(Regularizer):

    def __init__(
            self, 
            weight: float, 
            stopping_idx: List[int]
    ) -> None:
        """
        input:
        - stopping_idx: a list of indices of the stopping words, which are not calculated in the WMMD
        """
        self.stopping_idx = torch.tensor(stopping_idx)
        self.weight = weight

    def __call__(self, ids, counts, labels, **kwargs) -> torch.Tensor:
        """
        Compute word-level Maximum Mean Discrepancy using mini-batch data.
        """
        # if there is only one label in the mini-batch, return 0
        if len(torch.unique(labels)) == 1:
            return torch.tensor(0, dtype=torch.float32)

        # get the unique vocabulary indices in the mini-batch
        ids_unique = torch.unique(ids)
        
        # remove the stopping words from the unique vocabulary indices
        ids_unique = ids_unique[~torch.isin(ids_unique, self.stopping_idx)]
        # print(ids_unique)

        # get the embedding of the ids_unique and calculate the square of Euclidean distance of each pair of word vectors
        embedding = self.vocab_embedding[ids_unique]
        dist = self._word_dist(embedding)

        # remove zero counts
        counts = counts[:, ids_unique]

        # get the unique labels in the mini-batch
        unique_labels = torch.unique(labels)
        K = len(unique_labels) # the number of unique labels in mini-batch data

        # split the counts tensor by labels
        split_counts = {label.item(): counts[labels == label] for label in unique_labels}

        # for each pair of labels, calculate the counting matrix of each pair of words: P_neg, P_pos, P_neg_pos
        wmmd_all = torch.tensor(0, dtype=torch.float32)
        for i, label_neg in enumerate(unique_labels):
            for j, label_pos in enumerate(unique_labels):
                if i < j:
                    wmmd = self._pair_wmmd(split_counts, label_neg, label_pos, dist)
                    # print(f'sum of P_neg: {torch.sum(P_neg)},  sum of P_pos: {torch.sum(P_pos)},  sum of P_neg_pos: {torch.sum(P_neg_pos)}')

                    # add the wMMD to the total wMMD
                    wmmd_all += wmmd

        return -self.weight * wmmd_all
    

    def _pair_wmmd(self, split_counts, label_neg, label_pos, dist):
        """
        Calculate the wMMD between the negative and positive samples.
        """
        counts_neg = split_counts[label_neg.item()]
        counts_pos = split_counts[label_pos.item()]
        P_neg, P_pos, P_neg_pos = self._pair_count_matrix(counts_neg, counts_pos)

        # When the sum of P_neg or P_pos is 0 (it implies that number of negative or positive samples is 1), set the wMMD to 0
        if torch.sum(P_neg) == 0 or torch.sum(P_pos) == 0:
            return torch.tensor(0, dtype=torch.float32)

        # set sigma2 to the weighted median of the distances
        P = P_neg + P_pos + 2 * P_neg_pos
        sigma2 = self._weighted_median(dist.flatten(), P.flatten()).detach()
        gamma = 1/(2 * sigma2)

        # calculate the gaussian kernel distance matrix 
        K_dist = torch.exp(-gamma * dist)
        
        # calculate the wMMD between the negative and positive samples
        wmmd = torch.sum(P_neg * K_dist) / torch.sum(P_neg) + torch.sum(P_pos * K_dist) / torch.sum(P_pos) - 2 * torch.sum(P_neg_pos * K_dist) / torch.sum(P_neg_pos)
        
        return wmmd

    def _pair_count_matrix(self, counts_neg, counts_pos):
        """
        Calculate the counting matrix of each pair of words: P_neg, P_pos, P_neg_pos.
        inputs:
        - counts_neg: a 2-D tensor of counts for negative samples
        - counts_pos: a 2-D tensor of counts for positive samples
        output:
        - P_neg: a 2-D tensor of the counting matrix of each pair of words for negative samples
        - P_pos: a 2-D tensor of the counting matrix of each pair of words for positive samples
        - P_neg_pos: a 2-D tensor of the counting matrix of each pair of words between negative and positive samples
        """
        counts_neg, counts_pos = counts_neg.float(), counts_pos.float()

        # get P_neg and P_pos
        tmp_neg = torch.matmul(torch.ones(1, counts_neg.size(0)), counts_neg)
        tmp_pos = torch.matmul(torch.ones(1, counts_pos.size(0)), counts_pos)
        P_neg = torch.matmul(torch.transpose(tmp_neg, 0, 1), tmp_neg) - torch.matmul(torch.transpose(counts_neg, 0, 1), counts_neg)
        P_pos = torch.matmul(torch.transpose(tmp_pos, 0, 1), tmp_pos) - torch.matmul(torch.transpose(counts_pos, 0, 1), counts_pos)

        # get P_neg_pos
        P_neg_pos = torch.matmul(torch.transpose(tmp_neg, 0, 1), tmp_pos)

        return P_neg, P_pos, P_neg_pos


    def _word_dist(self, embedding):
        """
        Calculate the square of Euclidean distance of each pair of word embeddings.
        inputs:
        - embedding: a 2-D tensor of word embeddings
        output:
        - dist: a 2-D tensor of the square of Euclidean distance of each pair of word embeddings
        """

        # Calculate the square of Euclidean distance of each paif of word embeddings
        dist = torch.pow(torch.cdist(embedding, embedding, p=2), 2)
        return dist

    def _weighted_median(self, values, counts):
        """
        Calculate the weighted median of a list of values with corresponding counts.
        inputs:
        - values: a 1-D tensor of values
        - counts: a 1-D tensor of counts
        output:
        - median_value: the weighted median of the values
        """
        # Remove zero counts
        values = values[counts > 0]
        counts = counts[counts > 0]

        # Sort the values tensor
        sorted_values, sorted_indices = torch.sort(values)

        # Compute the cumulative sum of the counts tensor
        cumulative_counts = torch.cumsum(counts[sorted_indices], dim=0)

        # Normalize the cumulative sum to get the cumulative distribution function (CDF)
        cdf = cumulative_counts.float() / cumulative_counts[-1]

        # Find the index where the CDF first exceeds 0.5. This is the exact index or the next index of the median
        median_index = torch.argmax((cdf > 0.5).float())

        # Get the median value
        if cumulative_counts[-1] % 2 == 0 and cdf[median_index-1] > (0.5 - 1/cumulative_counts[-1]):
            median_value = 0.5 * (sorted_values[median_index - 1] + sorted_values[median_index])
        else:
            median_value = sorted_values[median_index]

        return(median_value)



class SwMMD(wMMD):
    """
    Structured wMMD regularizer.
    """
    def __init__(self, weight: float, stopping_start_idx: int, stopping_end_idx: int) -> None:
        stopping_idx = np.arange(stopping_start_idx, stopping_end_idx)
        super().__init__(weight, stopping_idx)




class Bigram_wMMD(Regularizer):
    def __init__(self, weight: float, stopping_idx: List[int]) -> None:
        """
        input:
        - stopping_idx: a list of indices of the stopping words, which are not calculated in the Bigram_wMMD
        """
        self.weight = weight
        self.stopping_idx = stopping_idx

    def __call__(self, ids, labels, **kwargs) -> torch.Tensor:
        """
        Compute word-level Bigram Maximum Mean Discrepancy using mini-batch data, only for the case where each sample has two single words.
        inputs:
        - ids: a 2-D tensor of vocabulary indices
        - labels: a 1-D tensor of labels
        output:
        - bigram_wmmd: a scalar tensor of bigram wMMD
        """
        # separate ids to two groups based on labels
        ids_neg = ids[labels == 0]
        ids_pos = ids[labels == 1]

        # if there is only one label or no label in either group, return 0
        if len(ids_neg) <= 1 or len(ids_pos) <= 1:
            return torch.tensor(0, dtype=torch.float32)
        
        # get the distances between each pair of indice
        dist_neg, dist_pos, dist_neg_pos = self._bigram_dist(ids_neg, ids_pos)

        # get median value of square of the distances
        dist = torch.cat([dist_neg, dist_pos, dist_neg_pos, dist_neg_pos])
        sigma2 = torch.median(dist).detach()
        # sigma2 = 1
        # print(f"algo1 sigma2: {sigma2}")
        if sigma2 == 0:
            return torch.tensor(0, dtype=torch.float32)
        else:
            gamma = 1/(2 * sigma2)

        # apply gaussian kernel on square of the distances
        kernel_dist_neg, kernel_dist_pos, kernel_dist_neg_pos = torch.exp(-gamma * dist_neg), torch.exp(-gamma * dist_pos), torch.exp(-gamma * dist_neg_pos)
        bigram_wmmd = torch.mean(kernel_dist_neg) + torch.mean(kernel_dist_pos) - 2 * torch.mean(kernel_dist_neg_pos)
        # print(torch.mean(kernel_dist_neg), torch.mean(kernel_dist_pos), 2 * torch.mean(kernel_dist_neg_pos))

        return -self.weight * bigram_wmmd

    def _bigram_dist(self, ids_neg, ids_pos):
        """
        Calculate the square of Euclidean distance of each pair of bigram embeddings.
        inputs:
        - ids_neg: a 2-D tensor of indices of negative samples
        - ids_pos: a 2-D tensor of indices of positive samples
        output:
        - dist_neg: a 1-D tensor of the square of Euclidean distance of each pair of bigram embeddings for negative samples
        - dist_pos: a 1-D tensor of the square of Euclidean distance of each pair of bigram embeddings for positive samples 
        - dist_neg_pos: a 1-D tensor of square of Euclidean distance of each pair of bigram embeddings between negative and positive samples
        """

        # replicate ids by 2 except the first and the last elements, it is to help get the embedding of bigrams
        ids_neg_rep = torch.vstack([self._replicate_idx(sentence) for sentence in ids_neg])
        ids_pos_rep = torch.vstack([self._replicate_idx(sentence) for sentence in ids_pos])

        # create a sentence indicator for the indice
        group_neg = torch.arange(ids_neg.size(0)).repeat_interleave(ids_neg.size(1)-1)
        group_pos = torch.arange(ids_pos.size(0)).repeat_interleave(ids_pos.size(1)-1)

        # get a mask matrix whose element is 1 if the two indices are in the same sentence, otherwise 0
        mask_neg = (group_neg.unsqueeze(1) == group_neg.unsqueeze(0)).float()
        mask_pos = (group_pos.unsqueeze(1) == group_pos.unsqueeze(0)).float()

        # get embeddings for two groups of ids
        embedding_neg = self.vocab_embedding[ids_neg_rep]
        embedding_pos = self.vocab_embedding[ids_pos_rep]

        # transform the embedding tensors to 2-D tensors
        embedding_neg = embedding_neg.view(-1, embedding_neg.size(2) * 2)
        embedding_pos = embedding_pos.view(-1, embedding_pos.size(2) * 2)

        # Calculate the square of Euclidean distance of each paif of bigram embeddings
        dist_neg = torch.pow(torch.cdist(embedding_neg, embedding_neg, p=2), 2)
        dist_pos = torch.pow(torch.cdist(embedding_pos, embedding_pos, p=2), 2)
        dist_neg_pos = torch.pow(torch.cdist(embedding_neg, embedding_pos, p=2), 2)
        # print(dist_pos)

        # select non-same-sentence elements from dist_neg and dist_pos into two 1-D tensor
        dist_neg = dist_neg[mask_neg == 0].view(-1)
        dist_pos = dist_pos[mask_pos == 0].view(-1)
        
        # select all elements from dist_neg_pos into a 1-D tensor
        dist_neg_pos = dist_neg_pos.view(-1)

        return dist_neg, dist_pos, dist_neg_pos
    
    def _replicate_idx(self, idx):
        """
        Replcate the indices in idx by 2 except the first and the last elements.
        inputs:
        - idx: a 1-D tensor of indices
        output:
        - idx_rep: a 1-D tensor of replicated indices
        """
        n = len(idx)
        idx_rep = torch.cat([idx[0].view(1), idx[1:-1].repeat_interleave(2), idx[-1].view(1)])
        return idx_rep