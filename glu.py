import torch 
import torch.nn as nn 
from torch.nn import functional as F
import numpy as np 
import tiktoken 
import json 
import time 


## Setting up the hyperparameters

n_embd = 32
batch_size = 64
block_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dropout_layer = 0.2
eval_iters = 500 ## Each batch loss 
learning_rate = 3e-2
max_iters = 5000
eval_interval = 500


### Loading the data from the files
enc = tiktoken.get_encoding('gpt2')
vocab_size = enc.n_vocab ## This gives the total tokens used in the gpt2 encoder
print(vocab_size)
## Training and validaiton split
data = torch.tensor(np.memmap('../localgpt/data/train.bin', dtype = np.uint16, mode = 'r'))
n = int(0.9 * len(data))
train_data = data[:n] ## Loading the training data
validation_data = data[n:] ## Loading the validation data
print(len(train_data))


## Creating the batches 
def batch_creater(split):
    data = train_data if split == 'train' else validation_data
    index = torch.randint(len(data)- block_size, (batch_size, ))
    xb = torch.stack(data[i:i+block_size] for i in range(index))
    yb = torch.stack(data[i+1:i+block_size+1] for i in range(index))
    return xb.long(), yb.long()

### Starting the model 

class zalGpt(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, n_embd) ##Creating the embedding table 
        self.blocks = nn.Sequential(
            Block(4, n_embd),
            Block(4, n_embd),
            Block(4, n_embd)
        )