# Word-level Maximum Mean Discrepancy Regularization for Word Embedding

`wMMD` is a Python module for implementing word-level maximum mean discrepancy (wMMD) regularization.

## Installation

### Dependency

`wMMD` requires Python 3.9 + Python libraries:
```shell
pip install -r requirements.txt
```

### Source code

You can check the latest code with the command:
```
git clone https://github.com/youqiangao/wMMD.git
```

## Usage

To train a `CNN` model on the `ChileEarthquakeT1` dataset with the `wMMD` (weight = 10^3), `l1` (weight = 10^3) or `dropout` (dropout rate = 0.5), repsectively execute the following commands:
```shell
python main.py --dataset chile-earthquakeT1 --model cnn --regularization wmmd --weight-exponent 3
```
```shell
python main.py --dataset chile-earthquakeT1 --model cnn --regularization l1 --weight-exponent 3
```
```shell
python main.py --dataset chile-earthquakeT1 --model cnn --structure dropout --dropout-rate 0.5
```

To check more arguments of the commands, run 
```shell
python main.py --help
```

The `regularizer.py` file contains the implementation of the `wMMD` regularization. To train your models using `wMMD` regularization, simply include an additional line in your Python code.
```python
loss = loss - weight * wMMD(model.embedding.weight, stopping_idx)
```
The `weight` hyperparameter balances the objective loss function and regularization loss. The `stopping_idx` includes the indices you wish to exclude from the calculation of `wMMD` regularization. By default, it is set to `[0]`, representing the index for padding.

## Results on real datasets

+ Three different neural network architectures are considered: `GRU`, `BiLSTM`, and `CNN`. 
+ The effects of various embedding sizes are investigated with $r$ = 20, 50, 100, 200, 300, as well as one pre-trained embedding with a fixed size of 300.
+ For `L1` and `wMMD`, the optimal weight $\lambda$ is determined by a grid search over $\lbrace 10^{(s-16)/5}, s = 1, \dots, 36 \rbrace$. As for `dropout`, the best dropout rate is tuned by a grid search over $\lbrace 0.05s, s = 1, \dots, 19 \rbrace$. The prediction performance of all methods is measured by classification accuracy.

### Results on the `ChileEarthquakeT1` dataset

+ We report average classification accuracy and standard error (in parenthesis) of 20 replications on `ChileEarthquakeT1`. The highest performance in each setting is highlighted in **bold**.
+ The pretrained embedding denoted as `SBWC` can be found in [this link](https://github.com/dccuchile/spanish-word-embeddings).

| model       | embedding size | No regularization | L1            | dropout       | wMMD (our)          |
|-------------|----------------|-------------------|---------------|---------------|------------------------|
| GRU         | 20             | 76.00 (0.238)     | 76.75 (0.325) | 79.78 (0.264) | **80.63 (0.281)** |
|             | 50             | 76.29 (0.275)     | 76.62 (0.245) | 80.04 (0.243) | 80.16 (0.296)          |
|             | 100            | 75.67 (0.272)     | 76.33 (0.285) | 79.81 (0.210) | **80.50 (0.217)** |
|             | 200            | 75.40 (0.194)     | 76.30 (0.297) | 79.85 (0.240) | **80.43 (0.237)** |
|             | 300            | 75.02 (0.206)     | 76.22 (0.354) | 79.60 (0.255) | **80.28 (0.217)** |
| GRU+SBWC    | 300            | 78.39 (0.208)     | 79.18 (0.288) | 80.97 (0.259) | **81.25 (0.220)** |
| BiLSTM      | 20             | 76.75 (0.244)     | 77.10 (0.231) | 79.72 (0.230) | **80.61 (0.246)** |
|             | 50             | 77.02 (0.334)     | 77.23 (0.241) | 79.82 (0.233) | **80.48 (0.277)** |
|             | 100            | 76.93 (0.279)     | 77.18 (0.196) | 79.89 (0.247) | **80.56 (0.241)** |
|             | 200            | 76.33 (0.227)     | 76.93 (0.285) | 80.06 (0.234) | **80.34 (0.247)** |
|             | 300            | 76.30 (0.167)     | 76.86 (0.299) | 79.39 (0.290) | **80.35 (0.250)** |
| BiLSTM+SBWC | 300            | 78.88 (0.256)     | 79.49 (0.350) | 81.13 (0.271) | 81.18 (0.262)          |
| CNN         | 20             | 77.52 (0.235)     | 78.22 (0.245) | 78.24 (0.249) | **80.43 (0.311)** |
|             | 50             | 77.29 (0.225)     | 77.29 (0.225) | 78.01 (0.255) | **80.16 (0.296)** |
|             | 100            | 76.50 (0.178)     | 77.38 (0.271) | 77.54 (0.211) | **79.51 (0.265)** |
|             | 200            | 76.48 (0.233)     | 77.26 (0.239) | 77.51 (0.207) | **79.83 (0.246)** |
|             | 300            | 75.99 (0.191)     | 77.03 (0.226) | 77.52 (0.207) | **79.77 (0.239)** |
| CNN+SBWC    | 300            | 78.61 (0.252)     | 78.64 (0.249) | 79.77 (0.318) | **80.25 (0.245)** |


## Results on the `BBC News` Dataset

+ Average classification accuracy and standard error (in parenthesis) is presented in the following table, based on 10 replications on the `BBC News` dataset. The highest performance in each setting is highlighted in **bold**.
+ The pretrained embedding denoted as `GoogleNews` can be found in [this link](https://github.com/RaRe-Technologies/gensim-data).

| model             | embedding size | No regularization | L1            | dropout       | wMMD (our)          |
|-------------------|----------------|-------------------|---------------|---------------|------------------------|
| GRU               | 20             | 86.21 (1.097)     | 88.27 (0.417) | 89.81 (0.963) | **94.98 (0.324)** |
|                   | 50             | 90.84 (1.117)     | 92.99 (0.419) | 93.16 (0.792) | **95.57 (0.266)** |
|                   | 100            | 90.35 (0.740)     | 93.83 (0.662) | 94.46 (0.438) | **95.80 (0.123)** |
|                   | 200            | 92.36 (0.412)     | 93.64 (0.367) | 95.07 (0.685) | **96.04 (0.153)** |
|                   | 300            | 92.63 (0.723)     | 94.07 (0.425) | 95.30 (0.220) | **96.10 (0.136)** |
| GRU+GoogleNews    | 300            | 93.49 (0.467)     | 94.44 (0.287) | 94.16 (0.278) | 94.48 (0.215)          |
| BiLSTM            | 20             | 89.06 (0.734)     | 90.56 (0.765) | 92.74 (0.323) | **94.75 (0.231)** |
|                   | 50             | 91.36 (1.186)     | 93.30 (0.626) | 94.75 (0.278) | **95.54 (0.237)** |
|                   | 100            | 92.09 (0.534)     | 93.72 (0.399) | 95.66 (0.302) | **95.80 (0.116)** |
|                   | 200            | 94.13 (0.297)     | 94.73 (0.235) | 95.72 (0.343) | **96.22 (0.162)** |
|                   | 300            | 93.53 (0.681)     | 94.98 (0.368) | 95.48 (0.194) | **96.19 (0.182)** |
| BiLSTM+GoogleNews | 300            | 92.34 (0.274)     | 92.98 (0.393) | 93.20 (0.400) | **94.34 (0.347)** |
| CNN               | 20             | 96.68 (0.113)     | 96.82 (0.151) | 96.90 (0.151) | **97.06 (0.088)** |
|                   | 50             | 97.21 (0.108)     | 97.22 (0.120) | 97.23 (0.135) | 97.26 (0.125) |
|                   | 100            | 97.25 (0.119)     | 97.25 (0.118) | 97.27 (0.142) | **97.42 (0.096)** |
|                   | 200            | 97.19 (0.144)     | 97.22 (0.110) | 97.25 (0.131) | **97.37 (0.095)** |
|                   | 300            | 97.08 (0.177)     | 97.26 (0.161) | 97.18 (0.110) | 97.27 (0.129)          |
| CNN+GoogleNews    | 300            | 96.76 (0.137)     | 96.81 (0.149) | 96.77 (0.141) | **96.91 (0.141)** |



### Tuning with no tears

+ The following figure displays average classification accuracy and 95% confidence interval of `CNN` trained on `ChileEarthquakeT1` using various hyperparameter values. The embedding size is 20. The gray dashed line represents the average accuracy with no regularization.
+ In constrst to `dropout` and `L1`, the performance with `wMMD` regularization initially increases and then stabilizes as the weight increases. Furthermore, `wMMD` exhibits a broader range of optimal weights compared to L1. **These properties significantly mitigate the hyperparameter-tuning issue**. It is noteworthy that similar trends are observed for other models and embedding sizes on both datasets in the conducted experiments.

![tuning-visualization](https://user-images.githubusercontent.com/26051979/267685125-832cffe8-2ade-48d5-b756-9ef2ed25a30c.png)

<!-- test  -->