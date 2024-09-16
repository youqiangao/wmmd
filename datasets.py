import numpy as np
import pandas as pd
import os
from scipy.stats import betabinom
import torch 
from typing import List, Tuple
import pickle
from preprocess_utils import Preprocessor
import logging 

logger = logging.getLogger(__name__)

class SimDataset1(torch.utils.data.Dataset):
    def __init__(
        self, 
        sample_size: int,
        vocab_num: int, 
        word_size: int, 
        dist0_alpha: float, 
        dist0_beta: float, 
        dist1_alpha: float, 
        dist1_beta: float,
        **kwargs,
    ) -> None:
        super(SimDataset1, self).__init__()
        self.sample_size = sample_size
        self.vocab_num = vocab_num
        self.word_size = word_size
        self.dist0_alpha = dist0_alpha
        self.dist0_beta = dist0_beta
        self.dist1_alpha = dist1_alpha
        self.dist1_beta = dist1_beta
        self.dataset = self._generate()

    def __getitem__(self, index: List) -> List:
        return self.dataset[index]
    
    def __len__(self) -> int:
        return len(self.dataset)

    def _generate(self) -> List[Tuple]:
        labels = np.random.choice(np.arange(2), size = (self.sample_size,))
        rv0 = betabinom(self.vocab_num - 1, self.dist0_alpha, self.dist0_beta)
        rv1 = betabinom(self.vocab_num - 1, self.dist1_alpha, self.dist1_beta)
        vocab_id = []
        for idx in labels:
            if idx == 0:
                vocab_id.append(rv0.rvs(self.word_size))
            else:
                vocab_id.append(rv1.rvs(self.word_size))
        labels = np.int64(labels)
        vocab_id = np.array(vocab_id)
        
        # get the counts of each word in the sentence
        counts = np.vstack([np.bincount(sentence, minlength = self.vocab_num) for sentence in vocab_id])

        data = [(id, label, count) for id, label, count in zip(vocab_id, labels, counts)]
        return data


class SimDataset2(torch.utils.data.Dataset):
    """
    Generate data for the simulation. Specifically, Y ~ Bernoulli(0.5), 
    (X|Y = 0) = r * beta-binomial(true_vocab_num-1, alpha0, beta0) + (1 - r) * uniform(true_vocab_num, total_vocab_num),
    (X|Y = 1) = r * beta-binomial(true_vocab_num-1, alpha1, beta1) + (1 - r) * uniform(true_vocab_num, total_vocab_num),
    """
    def __init__(
        self, 
        sample_size: int,
        true_vocab_num: int,
        vocab_num: int, 
        word_size: int, 
        prob_r: float, 
        dist0_alpha: float, 
        dist0_beta: float, 
        dist1_alpha: float, 
        dist1_beta: float,
        **kwrags,
    ) -> None:
        super(SimDataset2, self).__init__()
        self.sample_size = sample_size
        self.true_vocab_num = true_vocab_num
        self.vocab_num = vocab_num
        self.word_size = word_size
        self.prob_r = prob_r
        self.dist0_alpha = dist0_alpha
        self.dist0_beta = dist0_beta
        self.dist1_alpha = dist1_alpha
        self.dist1_beta = dist1_beta
        self.dataset = self._generate()

    def __getitem__(self, index: List) -> List:
        return self.dataset[index]
    
    def __len__(self) -> int:
        return len(self.dataset)

    def _generate(self) -> List[Tuple]:
        labels = np.random.choice(np.arange(2), size = (self.sample_size,))
        rv0 = betabinom(self.true_vocab_num - 1, self.dist0_alpha, self.dist0_beta)
        rv1 = betabinom(self.true_vocab_num - 1, self.dist1_alpha, self.dist1_beta)
        vocab_id = []
        for i, label in enumerate(labels):
            if label == 0:
                x0 = rv0.rvs(self.word_size)
                x2 = np.random.randint(self.true_vocab_num, self.vocab_num, self.word_size)
                x = [tmp0 if np.random.uniform(0, 1, size = 1) <= self.prob_r else tmp2 for tmp0, tmp2 in zip(x0, x2)]
                vocab_id.append(x)
            else:
                x1 = rv1.rvs(self.word_size)
                x2 = np.random.randint(self.true_vocab_num, self.vocab_num, self.word_size)
                x = [tmp1 if np.random.uniform(0, 1, size = 1) <= self.prob_r else tmp2 for tmp1, tmp2 in zip(x1, x2)]
                vocab_id.append(x)
        labels = np.int64(labels)
        vocab_id = np.array(vocab_id)
        
        # get the counts of each word in the sentence
        counts = np.vstack([np.bincount(sentence, minlength = self.vocab_num) for sentence in vocab_id])
        
        data = [(id, label, count) for id, label, count in zip(vocab_id, labels, counts)]
        return data


class SimDataset3(torch.utils.data.Dataset):
    """
    Generate data for the simulation. Specifically, Y ~ Bernoulli(0.5), 
    (X|Y = 0) = [X_1, X_2] and (X|Y = 1) = [X_2, X_1]. X_1, X_2 follow from different distributions. 
    """
    def __init__(
        self, 
        sample_size: int,
        vocab_num: int, 
        dist0_alpha: float, 
        dist0_beta: float, 
        dist1_alpha: float, 
        dist1_beta: float,
        **kwrags,
    ) -> None:
        super(SimDataset3, self).__init__()
        self.sample_size = sample_size
        self.vocab_num = vocab_num
        self.dist0_alpha = dist0_alpha
        self.dist0_beta = dist0_beta
        self.dist1_alpha = dist1_alpha
        self.dist1_beta = dist1_beta
        self.dataset = self._generate()

    def __getitem__(self, index: List) -> List:
        return self.dataset[index]
    
    def __len__(self) -> int:
        return len(self.dataset)

    def _generate(self) -> List[Tuple]:
        labels = np.random.choice(np.arange(2), size = (self.sample_size,))
        x1 = betabinom(self.vocab_num - 1, self.dist0_alpha, self.dist0_beta).rvs(size = self.sample_size)
        x2 = betabinom(self.vocab_num - 1, self.dist1_alpha, self.dist1_beta).rvs(size = self.sample_size)
        vocab_id = []
        for i, label in enumerate(labels):
            if label == 0:
                vocab_id.append([x1[i], x2[i]])
            else:
                vocab_id.append([x2[i], x1[i]])
        labels = np.int64(labels)
        vocab_id = np.array(vocab_id)
        
        # get the counts of each word in the sentence
        counts = np.vstack([np.bincount(sentence, minlength = self.vocab_num) for sentence in vocab_id])
        
        data = [(id, label, count) for id, label, count in zip(vocab_id, labels, counts)]
        return data


class CET1Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repreprocess: bool = False,
        **kwargs,
    ) -> None:
        super(CET1Dataset, self).__init__()
        self.repreprocess = repreprocess
        self.dataset = self._preprocess()
    
    def __getitem__(self, index: List) -> List:
        return self.dataset[index]
    
    def __len__(self) -> int:
        return len(self.dataset)

    def _preprocess(self) -> List[Tuple]:
        # check if the data is already preprocessed
        if not os.path.exists('tmp/datasets/'):
            os.makedirs('tmp/datasets/')
        if os.path.exists('tmp/datasets/ce-t1.pkl') and not self.repreprocess:
            with open('tmp/datasets/ce-t1.pkl', 'rb') as f:
                return pickle.load(f)
        
        # load the raw data and preprocess
        df = pd.read_csv('datasets/CE-T1/dataset.csv')
        labels = df['value']
        labels = [1 if (label == True) else 0 for label in labels]
        labels = np.array(labels, dtype=np.int64)
        texts = df['texto']
        preprocessor = Preprocessor(language='spanish', max_len=32, min_freq=5)
        vocab_id, vocab_num = preprocessor(texts)
        vocab_id = np.array(vocab_id)
        counts = np.vstack([np.bincount(sentence, minlength = vocab_num) for sentence in vocab_id])
        data = [(id, label, count) for id, label, count in zip(vocab_id, labels, counts)]

        with open('tmp/datasets/ce-t1.pkl', 'wb') as f:
            pickle.dump(data, f)
        return data

class BBCNewsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repreprocess: bool = False,
        **kwargs,
    ) -> None:
        super(BBCNewsDataset, self).__init__()
        self.repreprocess = repreprocess
        self.dataset = self._preprocess()
    
    def __getitem__(self, index: List) -> List:
        return self.dataset[index]
    
    def __len__(self) -> int:
        return len(self.dataset)

    def _preprocess(self) -> List[Tuple]:
        # check if the data is already preprocessed
        if not os.path.exists('tmp/datasets/'):
            os.makedirs('tmp/datasets/')
        if os.path.exists('tmp/datasets/bbc-news.pkl') and not self.repreprocess:
            with open('tmp/datasets/bbc-news.pkl', 'rb') as f:
                return pickle.load(f)
        
        # load the raw data and preprocess
        self._extract()
        with open('tmp/datasets/raw-bbc-news.pkl', 'rb') as f:
            df = pickle.load(f)
        labels = [_[1] for _ in df]
        # transform the labels to integers
        label2id = {label: idx for idx, label in enumerate(set(labels))}
        labels = [label2id[label] for label in labels]
        labels = np.array(labels, dtype=np.int64)

        texts = [_[0] for _ in df]
        preprocessor = Preprocessor(language='english', max_len=512, min_freq=10)
        vocab_id, vocab_num = preprocessor(texts)
        vocab_id = np.array(vocab_id)
        counts = np.vstack([np.bincount(sentence, minlength = vocab_num) for sentence in vocab_id])
        data = [(id, label, count) for id, label, count in zip(vocab_id, labels, counts)]
        logger.info(f"Number of samples: {len(data)}")

        with open('tmp/datasets/bbc-news.pkl', 'wb') as f:
            pickle.dump(data, f)
        return data

    def _extract(self, data_dir: str = 'datasets/BBC-News/dataset') -> None:
        # base_dir = os.path.abspath(os.path.dirname('.'))
        # data_dir = os.path.join(base_dir, data_path)
        if not os.path.exists(data_dir):
            logger.error(f'Directory {data_dir} does not exist.', stack_info=True, exc_info=True)
        if os.path.exists('tmp/dataset/raw-bbc-news.pkl'):
            return None

        logger.info(f'Extracting the dataset from {data_dir}...')
        walk = os.walk(data_dir)
        data = []
        for path, _, files in walk:
            for filename in files:
                if 'txt' in filename:
                    # file_path = os.path.join(base_dir, path, filename)
                    file_path = os.path.join(path, filename)
                    label = os.path.dirname(file_path).split('/')[-1]
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            text = ' '.join(f.readlines())
                        except:
                            logger.error(f'Error reading file {file_path}.', stack_info=True, exc_info=True)
                    data.append((text, label))  
            
        logger.info(f'The number of samples: {len(data)}.')
        with open('tmp/datasets/raw-bbc-news.pkl', 'wb') as f:
            pickle.dump(data, f)
        
    def _label2id(self, labels: List[str]) -> List[int]:
        label2id = {label: idx for idx, label in enumerate(set(labels))}
        return [label2id[label] for label in labels]
