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
dropout_layer = 0.2
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

### Creating a Self Attention Head Block 
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size) ## Key defines or eexplains what the token has to oofer or how can it be seen 
        self.value = nn.Linear(n_embd, head_size) ## Value defines what information that the token has 
        self.query = nn.Linear(n_embd, head_size) ## Query defines on the need or want or matching element
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size))) ## Creates a 256 X 256 buffeer that is not trainable but stores a pattern to prevent the model seing future information while predicting 
        self.dropout = nn.Dropout(dropout_layer)
    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x) ## 256 X 32 (x) * 32 X 8 -> 256 X 8 
        q = self.query(x) ## Shape = 256 X 8 
        v = self.value(x) ## Shape = 256 X 8
        wei = q @ k.transpose(-2,-1) * self.head_size ** -0.5 ## Normalizing the resulting matrix of 256 X 256 with the help of head_size
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) ## This fills up the weight matrix created with the respect to the buffer 
        wei = F.softmax(wei, dim = -1) ## Taking out the probability from the whole matrix
        out = wei @ v ## 256 X 256 @ 256 X 8 => 256 X 8
        



## Creating a Multi Head Attention Block
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size)] for _ in range(n_head))


## Lets create a transformer block 
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)

## Creating a bigram language model 
class GalGPT(nn.Module):
    def __init__(self):
        ## Creating an embedding table 
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) ## This is what stores the position of the given token 
        self.blocks = nn.Sequential(
            Block()
        )

