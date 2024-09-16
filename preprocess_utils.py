from typing import List
import re
import string
from nltk import word_tokenize, SnowballStemmer
from nltk.corpus import stopwords
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class Preprocessor(object):
    def __init__(self, language: str, max_len: int, min_freq: int = 1) -> None:
        self.language = language
        self.max_len = max_len
        self.min_freq = min_freq
        self.stop_words = self._stop_words()
        self.stemmer = SnowballStemmer(language)

    def __call__(self, texts: List[str]) -> List[List[int]]:
        # start preprocessing
        logger.info("Start preprocessing...")
        texts = [self._clean(text) for text in texts]
        vocab_counter = VocabCounter(min_freq=self.min_freq)
        vocab_counter.update(texts)
        texts = vocab_counter.encode(texts)
        vocab_num = len(vocab_counter.vocab_dict)
        logger.info(f"Vocabulary size: {vocab_num}")
        texts = self._truncate_pad(texts)
        logger.info("Preprocessing done.")
        return texts, vocab_num

    def _clean(self, text: str) -> str:
        text = text.lower() # lowercase
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text) # remove punctuations
        text = re.sub('[%s]' % '\n', ' ', text) # remove new line
        text = re.sub(r'http\S+', '', text) # remove url 
        text = re.sub(r'[0-9]', ' ', text) # remove numbers
        text = ' '.join(self.stemmer.stem(word) for word in word_tokenize(text) if word not in self.stop_words) # remove stopwords 
        return text

    def _truncate_pad(self, texts: List[List[int]]) -> List[List[int]]:
        texts_new = []
        for text in texts:
            if len(text) > self.max_len:
                texts_new.append(text[:self.max_len])
            else:
                texts_new.append(text + [0] * (self.max_len - len(text)))
        return texts_new

    def _stop_words(self) -> List[str]:
        stop_words = stopwords.words(self.language)
        if self.language == 'english':
            pass
        elif self.language == 'spanish':
            stop_words += ['rt', 'http', 'com']
        return stop_words



class VocabCounter():
    def __init__(self, min_freq = 1):
        self.min_freq = min_freq
        self.vocab_dict = {} # word to index
    
    def update(self, texts: List[str]) -> None:
        counter = Counter()
        for text in texts:
            counter.update(word_tokenize(text))
        sorted_tuple = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.vocab_dict['<pad>'] = 0 # 0 is the index of <PAD> for padding 
        for idx, (word, freq) in enumerate(sorted_tuple):
            if freq >= self.min_freq:
                self.vocab_dict[word] = idx + 1 # index starts from 1 

    def encode(self, texts: List[str]) -> List[List[int]]:
        return [[self.vocab_dict[word] for word in word_tokenize(text) if word in self.vocab_dict] for text in texts]

    def update_encode(self, texts: List[str]) -> List[List[int]]:
        self.update(texts)
        return self.encode(texts)
    

