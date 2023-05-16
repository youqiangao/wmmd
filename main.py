import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR
import pickle
import os
import json
from datetime import datetime
import argparse
import logging
from models import BiLSTM, GRU, CNN
from utils import collate_fn, train_loop, test_loop

logging.basicConfig()   
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info(f"We are using device {device}.")

# arguments parsing
parser = argparse.ArgumentParser(description='Import hyperparameters.')
parser.add_argument("-d", "--dataset", type = str, required = True, help = "name of dataset")
parser.add_argument("-m", "--model", type = str, required = True, help = "name of model")
parser.add_argument("-ts", "--train-size", type = float, default=0.5, help = "training size")
parser.add_argument("-bs", "--batch-size", type = int, default=64, help = "batch size")
parser.add_argument("-es", "--embed-size", type = int, default=50, help = "embedding size")
parser.add_argument("-lr", "--learning-rate", type = float, default=1e-3, help = "learning rate")
parser.add_argument("-e", "--epochs", type = int, default=60, help = "number of epochs")
parser.add_argument("-r", "--regularization", type = str, help = "regularization in the loss function")
parser.add_argument("-we", "--weight-exponent", type = float, help = "determine weight in the loss function, equal to 10 to the power of the exponent")
parser.add_argument("-st", "--structure", type = str, help = "some speical layer in the model")
parser.add_argument("-dr", "--dropout-rate", type = float, help = "probability of an element to be zeroed")
parser.add_argument("-s", "--seed", type = int, default=1, help = "random seed")
parser.add_argument("-o", "--output-dir", type = str, help = "the directory for the output")
parser.add_argument("-en", "--embed-name", type = str, help = "the name of pretrained embeddding")
args = parser.parse_args()

DATASETS = ["bbc", "chile-earthquakeT1"]
args.dataset = args.dataset
if args.dataset not in DATASETS:
    raise ValueError(f"dataset should be one of {DATASETS}.")
label_num = {
    "bbc": 5,
    "chile-earthquakeT1": 2
}.get(args.dataset)

PRETRAINED_EMBEDDINGS = ["word2vec-google-news-300"]
if args.embed_name is not None:
    args.embed_name = args.embed_name.lower() 
    args.embed_size = 300
    if args.embed_name not in PRETRAINED_EMBEDDINGS:
        raise ValueError(f"embedding-name should be one of {PRETRAINED_EMBEDDINGS}.")

MODELS = ["bilstm", "gru", "cnn"]
args.model = args.model.lower()
if args.model not in MODELS:
    raise ValueError(f"model should be one of {MODELS}.")

REGULARIZATIONS = ["l1", "wmmd"]
if args.regularization is None:
    args.weight = 0
else:
    args.regularization = args.regularization.lower()
    if args.regularization not in REGULARIZATIONS:
        raise ValueError(f"regularization should be one of {REGULARIZATIONS}.")
    if args.weight_exponent is not None:
        args.weight = 10 ** args.weight_exponent
    else: 
        raise ValueError("for the provided regularization, weight-exponent need to be specified.")

STRUCTURES = ["dropout"]
if args.structure is None:
    args.dropout_rate = None
else: 
    args.structure = args.structure.lower()
    if args.structure not in STRUCTURES:
        raise ValueError(f"--structure needs to be one of {STRUCTURES}.")
    if args.dropout_rate is None:
        raise ValueError("--dropout-rate needs to be provided for dropout regularization.")

if args.output_dir is None:
    args.output_dir = "./"

# random seed 
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True 


# load datasets
batch_size = args.batch_size
with open(os.path.join('datasets', args.dataset, 'dataset.pkl'), 'rb') as f:
    dataset = pickle.load(f)
num_word = dataset['meta']['num_word']
dataset = dataset['data']
logger.info(f'The total number of words is {num_word}')
train_dataset, test_dataset = train_test_split(dataset, train_size = args.train_size)
logger.info(f'The first element of train_dataset is {train_dataset[0]}')
train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, collate_fn = collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, collate_fn = collate_fn)


structure = {
    "dropout": {
        "structure": "dropout", 
        "dropout_rate": args.dropout_rate,
    },
    None: {
        "structure": None,
    }
}.get(args.structure)

model_name = args.model
if args.embed_name is None:
    if model_name == "bilstm":
        model = BiLSTM(label_num, num_word, args.embed_size, 0, **structure).to(device)
    elif model_name == "gru":
        model = GRU(label_num, num_word, args.embed_size, 0, **structure).to(device)
    elif model_name == "cnn":
        model = CNN(label_num, num_word, args.embed_size, 0, **structure).to(device)
# else:
#     with open(os.path.join('pretrained', args.embed_name + '.pkl'), 'rb') as f:
#         embedding = pickle.load(f)
#     if model_name == "bilstm":
#         model = BiLSTM.from_pretrained(label_num, embedding, 0, **structure).to(device)
#     elif model_name == "gru":
#         model = GRU.from_pretrained(label_num, embedding, 0, **structure).to(device)
#     elif model_name == "cnn":
#         model = CNN.from_pretrained(label_num, embedding, 0, **structure).to(device)

# training arguments
learning_rate = args.learning_rate
epochs = args.epochs
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = LinearLR(optimizer, start_factor=1., end_factor = 0., total_iters=epochs)
regularization = {
    None: {
        "regularization": None
    },
    "l1": {
        "regularization": "l1",
        "weight": args.weight
    },
    "wmmd": {
        "regularization": "wmmd",
        "weight": args.weight,
        "stopping_idx": [0]
    }
}.get(args.regularization)

# training
now = datetime.now()
start_time = now.strftime("%H:%M:%S")

for t in range(epochs):
    logger.info(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer, device, **regularization)
    acc = test_loop(test_dataloader, model, device)
    lr_scheduler.step()
logger.info("Done!")


# print start time and end time
now = datetime.now()
end_time = now.strftime("%H:%M:%S")
logger.info(f"Start training time: {start_time}")
logger.info(f"End training time: {end_time}")

# save the result to .json
result = vars(args)
result["accuracy"] = acc
with open(os.path.join(args.output_dir, 'result.json'), 'w') as outfile:
    json.dump(result, outfile)