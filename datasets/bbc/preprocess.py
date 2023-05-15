import numpy as np
import pandas as pd
from collections import Counter, OrderedDict
import re
import string
from nltk import word_tokenize, SnowballStemmer
from nltk.corpus import stopwords
import pickle
import logging

logging.basicConfig()
logger= logging.getLogger()
logger.setLevel(logging.INFO) 

# loading dataset
logging.info('Loading dataset.')
dataset = pd.read_csv('dataset.csv')
label2id = {
    'business': 0,
    'entertainment': 1,
    'politics': 2,
    'sport':3,
    'tech':4
}
texts = dataset['text'].to_list()
labels = dataset['label'].to_list()
labels = [label2id[label] for label in labels]

texts = np.array(texts)
labels = np.array(labels)


# pre-processing (clean + create dictionary + indexing + padding/truncation)
logger.info('Dataset has been loaded and preprocess starts.')
stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')

def clean(text):
    text = text.lower()
    text = re.sub(r"\'s", ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text) # remove punctuations
    text = re.sub('[%s]' % '\n', ' ', text)
    text = re.sub(r'[0-9]', ' ', text) # remove numbers
    text = ' '.join(stemmer.stem(word) for word in word_tokenize(text) if word not in stop_words) # remove stopwords 
    return text

class Vocab():
    def __init__(self, word_dict, min_freq = 1):
        self.index = 0
        self.dict = {}
        self._add('<PAD>') # 0 is always the index of <PAD>
        for (word, count) in word_dict.items():
            if count >= min_freq:
                self._add(word)
        
    def _add(self, word):
        self.dict[word] = self.index
        self.index += 1
    
    def __call__(self, words):
        dict_ = self.dict
        ids = [dict_[word] for word in words if word in dict_]
        return ids  
    
    def get_words(self):
        return list(self.dict.keys())

class Aligner():
    def __init__(self, padding_idx, max_len):
        self.padding_idx = padding_idx
        self.max_len = max_len
    
    def __call__(self, words):
        max_len = self.max_len
        pad_idx = self.padding_idx
        if len(words) <= max_len:
            words = words + [pad_idx] * (max_len - len(words)) # padding
        else:
            words = words[0:max_len] # truncation
        return words
    
# clean texts
clean_texts = [clean(instance) for instance in texts]

# create dictionary, exclude words with small frenquency
words_list = [word_tokenize(text) for text in clean_texts]
counter = Counter()
for words in words_list:
    counter.update(words)
sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
ordered_dict = OrderedDict(sorted_by_freq_tuples)
vocab = Vocab(ordered_dict, min_freq = 10)

# indexing
ids_list = [vocab(words) for words in words_list]

# truncation and padding
aligner = Aligner(padding_idx = 0, max_len = 512)
ids_list = [aligner(ids) for ids in ids_list]

# save the dataset after pre-processing, and pretrained_embedding
dataset = [instance for instance in zip(ids_list, labels)]
meta = {'num_word': len(vocab.get_words())}
dataset = {'data': dataset, 'meta': meta}

with open('dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)

logger.info('Preprocess has been completed.')

logger.info(f'The number of samples: {len(labels)}')
logger.info(f'The number of vocabularies in the dictionary is {len(vocab.get_words())}.')