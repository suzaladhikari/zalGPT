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


### Rotating frequency creater with the angular value of cos and sin 
def create_cache(block_size, head_size, theta = 10000.0):
    roatating_frequency = 1.0 / (theta ** (torch.arange(0,head_size,2).float() / head_size))
    blocks = torch.arange(block_size).float()
    updated_frequency = torch.outer(blocks, roatating_frequency) ## This is the product 
    return updated_frequency.cos(), updated_frequency.sin() ## Returning sin and cos values


def apply_rope_cache(x, sin, cos):
    T = x.shape[1]
    x1 = x[:, : , 0::2] ## picking the even dimensions
    x2 = x[:, :, 1::2] ## picking the odd dimensions
    cos, sin = cos[:T], sin[:T]
    rot1 = cos * x1 - x2 * sin 
    rot2 = sin * x1 + x2 * cos 
    return torch.stack([rot1, rot2], dim = -1).flatten(-2) ## Changing the shape back to the original(n_embd, head_size)
### Head 
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size)
        self.query = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)
        cos, sin = create_cache(block_size, head_size) ## Applying rope 
        self.register_buffer('cos', cos, persistent=False)
        self.register_buffer('sin', sin, persistent=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout_layer)

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        rotated_k = apply_rope_cache(k , self.sin, self.cos)
        rotated_q = apply_rope_cache(q, self.sin, self.cos)
        
### Multiple Head Attention 
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads =  nn.ModuleList(Head(head_size) for _ in range(n_heads))

class Block(nn.Module):
    def __init__(self, n_heads, n_embd):
        super().__init__()
        head_size = n_embd // n_heads
        self.heads = MultiHeadAttention(n_heads, head_size)


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