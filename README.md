# Word-level Maximum Mean Discrepancy Regularization for Embedding-based Neural Networks in NLP

`wMMD` is a Python module for implementing word-level Maximum Mean Discrepancy Regularization.

## Installation

### Dependency

`wMMD` requires Python 3.9 + Python libraries:
```shell
pip install -r requirements.txt
```

### Source code

You can check the latest code with the command:
```
git clone https://github.com/youqiangao/word-level-Maximum-Mean-Discrepancy.git
```

## Usage

To run the `CNN` model on `BBC` dataset with the `wMMD` regularization (weight = 10^2), use 
```shell
python main.py --dataset bbc --model cnn --regularization wmmd -we 2 
```

To check more arguments of the script, use 
```shell
python main.py --help
```

## Real aaplication results

+ Three different neural network architectures are considered: `GRU`, and `BiLSTM`, `CNN`. The detailed model structures are stored in [models.py](https://github.com/youqiangao/word-level-Maximum-Mean-Discrepancy/blob/main/models.py).
+ We consider different embedding sizes with $r$ = 20, 50, 100, 200, 300, and one pre-trained embeddings with sizes of 300 from Word2Vec (WV).
+ For `L1` and `wMMD`, the optimal weight $\lambda$ is determined by a grid search over $\{10^{(s-16)/5}, s = 1, \dots, 36\}$. For `dropout`, the best dropout rate is tuned by a grid search over $\{ 0.05s, s = 1, \dots, 19\}$. The prediction performance of all methods is measured by classification accuracy.

### Results on `ChileEarthquakeT1` dataset

+ Average classification accuracy and standard error (in parenthesis) of 20 replications on `ChileEarthquakeT1`. The highest performance in each setting is highlighted in **bold**.
+ The pretrained embedding from `Word2Vec` used for `ChileEarthquakeT1` can be found in [this link](https://github.com/dccuchile/spanish-word-embeddings).

| model  | embedding size | No regularization | L1            | dropout       | wMMD (our)          |
|--------|----------------|-------------------|---------------|---------------|------------------------|
| GRU    | 20             | 76.00 (0.238)     | 76.75 (0.325) | 79.78 (0.264) | **80.63 (0.281)** |
|        | 50             | 76.29 (0.275)     | 76.62 (0.245) | 80.04 (0.243) | **80.16 (0.296)** |
|        | 100            | 75.67 (0.272)     | 76.33 (0.285) | 79.81 (0.210) | **80.50 (0.217)** |
|        | 200            | 75.40 (0.194)     | 76.30 (0.297) | 79.85 (0.240) | **80.43 (0.237)** |
|        | 300            | 75.02 (0.206)     | 76.22 (0.354) | 79.60 (0.255) | **80.28 (0.217)** |
|        | 300 (WV) | 78.39 (0.208)     | 79.18 (0.288) | 80.97 (0.259) | **81.25 (0.220)** |
| BiLSTM | 20             | 76.75 (0.244)     | 77.10 (0.231) | 79.72 (0.230) | **80.61 (0.246)** |
|        | 50             | 77.02 (0.334)     | 77.23 (0.241) | 79.82 (0.233) | **80.48 (0.277)** |
|        | 100            | 76.93 (0.279)     | 77.18 (0.196) | 79.89 (0.247) | **80.56 (0.241)** |
|        | 200            | 76.33 (0.227)     | 76.93 (0.285) | 80.06 (0.234) | **80.34 (0.247)** |
|        | 300            | 76.30 (0.167)     | 76.86 (0.299) | 79.39 (0.290) | **80.35 (0.250)** |
|        | 300 (WV) | 78.88 (0.256)     | 79.49 (0.350) | 81.13 (0.271) | 81.18 (0.262)          |
| CNN    | 20             | 77.52 (0.235)     | 78.22 (0.245) | 78.24 (0.249) | **80.43 (0.311)** |
|        | 50             | 77.29 (0.225)     | 77.29 (0.225) | 78.01 (0.255) | **80.16 (0.296)** |
|        | 100            | 76.50 (0.178)     | 77.38 (0.271) | 77.54 (0.211) | **79.51 (0.265)** |
|        | 200            | 76.48 (0.233)     | 77.26 (0.239) | 77.51 (0.207) | **79.83 (0.246)** |
|        | 300            | 75.99 (0.191)     | 77.03 (0.226) | 77.52 (0.207) | **79.77 (0.239)** |
|        | 300 (WV) | 78.61 (0.252)     | 78.64 (0.249) | 79.77 (0.318) | **80.25 (0.245)** |

## Results on `BBC News` Dataset

+ Average classification accuracy and standard error (in parenthesis) of 20 replications on `BBC News`. The highest performance in each setting is highlighted in bold.
+ The pretrained embedding from `Word2Vec` used for `BBC News` can be found in [this link](https://github.com/RaRe-Technologies/gensim-data).

| model  |   | embedding size | No regularization | L1            | dropout       | wMMD (our)          |
|--------|---|----------------|-------------------|---------------|---------------|------------------------|
| GRU    |   | 20             | 76.00 (0.238)     | 76.75 (0.325) | 79.78 (0.264) | **80.63 (0.281)** |
|        |   | 50             | 76.29 (0.275)     | 76.62 (0.245) | 80.04 (0.243) | **80.16 (0.296)** |
|        |   | 100            | 75.67 (0.272)     | 76.33 (0.285) | 79.81 (0.210) | **80.50 (0.217)** |
|        |   | 200            | 75.40 (0.194)     | 76.30 (0.297) | 79.85 (0.240) | **80.43 (0.237)** |
|        |   | 300            | 75.02 (0.206)     | 76.22 (0.354) | 79.60 (0.255) | **80.28 (0.217)** |
|        |   | 300 (WV) | 78.39 (0.208)     | 79.18 (0.288) | 80.97 (0.259) | **81.25 (0.220)** |
| BiLSTM |   | 20             | 76.75 (0.244)     | 77.10 (0.231) | 79.72 (0.230) | **80.61 (0.246)** |
|        |   | 50             | 77.02 (0.334)     | 77.23 (0.241) | 79.82 (0.233) | **80.48 (0.277)** |
|        |   | 100            | 76.93 (0.279)     | 77.18 (0.196) | 79.89 (0.247) | **80.56 (0.241)** |
|        |   | 200            | 76.33 (0.227)     | 76.93 (0.285) | 80.06 (0.234) | **80.34 (0.247)** |
|        |   | 300            | 76.30 (0.167)     | 76.86 (0.299) | 79.39 (0.290) | **80.35 (0.250)** |
|        |   | 300 (WV) | 78.88 (0.256)     | 79.49 (0.350) | 81.13 (0.271) | 81.18 (0.262)          |
| CNN    |   | 20             | 77.52 (0.235)     | 78.22 (0.245) | 78.24 (0.249) | **80.43 (0.311)** |
|        |   | 50             | 77.29 (0.225)     | 77.29 (0.225) | 78.01 (0.255) | **80.16 (0.296)** |
|        |   | 100            | 76.50 (0.178)     | 77.38 (0.271) | 77.54 (0.211) | **79.51 (0.265)** |
|        |   | 200            | 76.48 (0.233)     | 77.26 (0.239) | 77.51 (0.207) | **79.83 (0.246)** |
|        |   | 300            | 75.99 (0.191)     | 77.03 (0.226) | 77.52 (0.207) | **79.77 (0.239)** |
|        |   | 300 (WV) | 78.61 (0.252)     | 78.64 (0.249) | 79.77 (0.318) | **80.25 (0.245)** |

### Tuning with no tears

Average classification accuracy and 95% confidence interval of `CNN` with an embedding dimension of 20 trained on `ChileEarthquakeT1` using various hyperparameters. The grey dashed line represents the average accuracy of `CNN` with no regularization. Line segments with average accuracy below 70% are omitted for clarity. 

![tuning-visualization](https://github.com/youqiangao/word-level-Maximum-Mean-Discrepancy/assets/26051979/95ab9bdb-284a-4176-a37d-afcf6370cf51)