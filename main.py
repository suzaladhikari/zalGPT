import torch 
import torch.nn as nn 
from torch.nn import functional as F
import tiktoken
import json
import numpy as np

torch.manual_seed(455841)

### Setting up the hyperparameters
n_embd = 32
batch_size = 64
block_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
### Each time the model gets the data of total 16,384 tokens/step

### Extracting the key words
chars = set() 
with open("../localgpt/data/TinyStories-train.txt", "r", encoding="utf-8") as f:
    text = f.read()
    for each_line in f:
        chars.update(each_line)

chars = sorted(chars)

with open('../localgpt/data/vocab.json', 'w', encoding='utf-8') as f:
    json.dump(chars,f)

## Setting up the vocab_size 
vocab_size = len(chars)

## Setting up the encoder and decoder
enc = tiktoken.get_encoding('gpt2')

## Training and validaiton split
data = torch.tensor(np.memmap('../localgpt/data/train.bin', dtype = np.uint16, mode = 'r'))
n = int(0.9 * len(data))
train_data = data[:n]
validation_data = data[n:]

## Creating the batches of data

def get_batch(split):
    # data = train_data if split == 'train' else validation_data if split == 'val' else test_data
    data = train_data if split == 'train' else validation_data 
    index = torch.randint(len(data)-block_size , (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in index])
    y = torch.stack([data[i+1:i+block_size+1] for i in index])
    return x,y

xb,yb = get_batch("train")
xb,yb = xb.to(device), yb.to(device)

## Creating a
