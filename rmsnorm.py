import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 
import tiktoken 

## Setting up the hyperparameters 
n_embd = 32
batch_size = 64
block_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dropout_layer = 0.2
eval_iters = 500 ## Each batch loss 
learning_rate = 3e-4
max_iters = 5000
eval_interval = 500

### Encoder and decoder 
enc = tiktoken.get_encoding('gpt2')
vocab_size = enc.n_vocab
print(vocab_size)

## Loading the data
data = torch.tensor(np.memmap('../localgpt/data/train.bin', dtype = np.uint16, mode = 'r'))
n = int(0.9 * len(data))
train_data = data[:n] ## Loading the training data
validation_data = data[n:] ## Loading the validation data
print(len(train_data))

### Creating batches
def creating_batches(split):
    data = train_data if split == 'train' else validation_data
    index = torch.rand(len(data)-block_size, (batch_size,)) ## Random integers of the size batch 
    xb = torch.stack([data[i:i+block_size] for i in index]) ## 64 X 256 
    yb = torch.stack([data[i+1: i+block_size+1] for i in index]) ## 64 x 256
    return xb.long(),yb.long()

def build_rope_cache(block_size, head_size, device, theta = 10000):
    rotating_frequency = 1.0 / (theta ** (torch.arange(0,head_size,2).float() / head_size))
    t = torch.arange(block_size).float()
    updated_frequency = torch.outer(rotating_frequency, t)
    return updated_frequency.cos(), updated_frequency.sin()

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)
        self.query = nn.Linear(n_embd, head_size)
        self.register_buffer('tril', torch.tril(block_size, block_size)) ## Creates a buffer which wont get modified during the gradient descent
        cos, sin = build_rope_cache(block_size, head_size, device) ## Rotating frequency and storing its value in terms of the sin and cos 
        self.register_buffer('cos', cos, persistent=False) ## 256 X 4
        self.register_buffer('sin', sin, persistent=False) ## 256 X 4 

    def forward(self, x):
        k = self.key(x)
        v = self.value(x)
        q = self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, number_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(number_heads)])
        

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd)
        )
    def forward(self,x):
        return self.net(x) ## 256 X32
## Creating block class 
class Block(nn.Module):
    def __init__(self, number_heads, n_embd):
        super().__init__()
        head_size = n_embd // number_heads
        self.sa_heads = MultiHeadAttention(number_heads, head_size)
        self.ffwd = FeedForward(n_embd)

### Creating the Model 
class zalGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(
            Block(4, n_embd),
            Block(4, n_embd),
            Block(4, n_embd)
        )

