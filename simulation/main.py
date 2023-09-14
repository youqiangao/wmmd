# import modules
import argparse
import json
import numpy as np
import torch 
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR
from data_gen import *
from models import LogisticRegression
from utils import *

# argument parser
parser = argparse.ArgumentParser(description='Import hyperparameters.')
parser.add_argument("--num-vocabs", type = int, default = 500, help = "total number of vocabularies")
parser.add_argument("--num-words", type = int, default = 10, help = "number of words in a sentence")
parser.add_argument("--sample-size", type = int, default = 200, help = "number of synthesized samples")
parser.add_argument("--embedding-size", type = int, default = 5, help = "size of embedding for each vocabularies")
parser.add_argument("--regularization", type = str, help = "regularization in the loss function")
parser.add_argument("--weight-exponent", type = float, help = "determine weight in the loss function, equal to 10 to the power of the exponent")
parser.add_argument("--structure", type = str, help = "some speical layer in the model")
parser.add_argument("--dropout-rate", type = float, help = "probability of an element to be zeroed")
parser.add_argument("--dist0-alpha", type = float, required = True, help = "value of alpha for beta distribution 0")
parser.add_argument("--dist1-alpha", type = float, default = 5, help = "value of alpha for beta distribution 1")
parser.add_argument("--seed", type = int, default = 1, help = "random seed")
parser.add_argument("--output-dir", type = str, help = "the directory for the output")
args = parser.parse_args()

device = torch.device("cpu")

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

# synthesize data
data_generator = DataGenerator(args.num_vocabs, args.num_words, args.dist0_alpha, args.dist0_alpha, args.dist1_alpha, args.dist1_alpha)
df = data_generator.generate(args.sample_size)
train_size = int(args.sample_size / 2)
test_size = args.sample_size - train_size
dataset = CustomDataset(df)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_dataloader = DataLoader(test_dataset, batch_size = 64, shuffle = True)

# training arguments
learning_rate = 1e-1
epochs = 30
loss_fn = nn.BCELoss()

# start training
structure = {
    "dropout": {
        "structure": "dropout", 
        "dropout_rate": args.dropout_rate,
    },
    None: {
        "structure": None,
    }
}.get(args.structure)

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
        "stopping_idx": []
    }
}.get(args.regularization)

model = LogisticRegression(args.num_vocabs, args.embedding_size, args.num_words, **structure) 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
lambda_fun = lambda epoch: 0.7 ** int(epoch / 5) # multiple the current learning rate by 0.7 for each 5 epochs.
lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda_fun)
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer, device, **regularization)
    if lr_scheduler:
        lr_scheduler.step()
    acc = test_loop(test_dataloader, model, device)
print("Done!")

result = vars(args)
result["accuracy"] = acc
with open(args.output_dir + '/result.json', 'w') as outfile:
    json.dump(result, outfile)