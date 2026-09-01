import torch 
import torch.nn as nn 
import numpy as np 
import tiktoken 
import json 

## Hyperparameters to be used 
learning_rate = 3e-4
batch_size = 64
n_embd = 32 
total_iterations = 2000
eval_iter = 20
block_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'

## Setting up the encoder and decoder 
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
def create_batches(split):
    data = train_data if split == 'train' else validation_data
    index = torch.randint(len(data) - block_size, (batch_size,))
    ## This return the random 64 different numbers from the len(data) - blocksize 
    x = torch.stack([data[i:i+block_size] for i in index]) ## 64 X 256 
    y = torch.stack([data[i+1:block_size+1] for i in index]) ## 64 X 256
    return x.long(), y.long()

xb, yb = create_batches("train")

## Creating a Head
class Head(nn.Module):
    def __init__(self):
        pass
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.sa_head = nn.Module([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim = -1) ## The learned data will be stacked next to each other creating 256 X 32 dimension matrix which will be used 
        out = self.proj(out) ## 256 X 32 @ 32 X 32 -> 256 X 32 
        return out  ## 256 X 32


## Creating a feed forward layer 
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super.__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4* n_embd),
            nn.ReLU(),
            nn.Linear(n_embd *4 , n_embd)
        )
    def forward(self, x):
        return self.net(x)
    
### Creating a transformer block 
class Block(nn.Module):
    def __init__(self, n_head, n_embd):
        super().__init__()
        head_size = n_embd // n_head
        self.sa_head = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x 
    
class zalGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.Sequential(
            Block(4, n_embd),
            Block(4, n_embd),
            Block(4, n_embd)
        )

model = zalGPT()
model.to(device)
## Creating a loss function 
@torch.no_grad()
def generate_loss():
    out = {}
    model.eval()
    for split in ['train','test']:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            xb, yb = create_batches(split)
            xb, yb = xb.to(device), yb.to(device)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out ## This returns the average of the 200 batches of losses for both training and validation



