import numpy as np
from collections import Counter, OrderedDict
import re
import string
import nltk
from nltk import word_tokenize, SnowballStemmer
from nltk.corpus import stopwords
import pandas as pd
import pickle
import logging

nltk.download('stopwords')
nltk.download('punkt')

logging.basicConfig()
logger= logging.getLogger()
logger.setLevel(logging.INFO) 

# loading dataset
logging.info('Loading dataset.')

dataset = pd.read_csv('dataset.csv')
texts = dataset['texto']
labels = dataset['value']
labels = [1 if (label == True) else 0 for label in labels]
texts = np.array(texts)
labels = np.array(labels)

# pre-processing (clean + create dictionary + indexing + padding/truncation)
logger.info('Dataset has been loaded and preprocess starts.')

stop_words = stopwords.words('spanish')
stop_words = stop_words  + ['rt', 'http', 'com']
stemmer = SnowballStemmer('spanish')

def clean(text):
    text = text.lower()
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text) # remove punctuations
    text = re.sub(r'http\S+', '', text) # remove url 
    text = re.sub(r'[0-9]', ' ', text)
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
            text_len = len(words)
            words = words + [pad_idx] * (max_len - len(words)) # padding
        else:
            text_len = self.max_len
            words = words[0:max_len] # truncation
        return words, text_len
    
# clean texts
clean_texts = [clean(instance) for instance in texts]

# create dictionary, exclude words with small frenquency
words_list = [word_tokenize(text) for text in clean_texts]
counter = Counter()
for words in words_list:
    counter.update(words)
sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
ordered_dict = OrderedDict(sorted_by_freq_tuples)
vocab = Vocab(ordered_dict, min_freq = 5)

# indexing
ids_list = [vocab(words) for words in words_list]

# truncation and padding
aligner = Aligner(padding_idx = 0, max_len = 32)
align_result = [aligner(ids) for ids in ids_list]
ids_list = [instance[0] for instance in align_result]

# save the dataset after pre-processing, and pretrained_embedding
dataset = [instance for instance in zip(ids_list, labels)]
meta = {'num_word': len(vocab.get_words())}
dataset = {'data': dataset, 'meta': meta}

logger.info('Preprocess has been completed.')

with open('dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)

logger.info(f'The number of samples: {len(labels)}')
logger.info(f'The number of vocabularies in the dictionary is {len(vocab.get_words())}.')