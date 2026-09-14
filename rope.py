import torch 
import torch.nn as nn 
from torch.nn import functional as F
import numpy as np 
import tiktoken 
import json 
import time 
## Hyperparameters to be used 


torch.manual_seed(455841)
n_embd = 64
batch_size = 64
block_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dropout_layer = 0.2
eval_iters = 1 ## Each batch loss 
learning_rate = 3e-2
max_iters = 1
eval_interval = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

### For RoPE setup \
def build_rope_cache(seq_len, head_size, device, theta = 10000.0):
    roatating_frequency = 1.0 / (theta ** (torch.arange(0,head_size,2).float() / head_size)) ## This gives the frequency that will be used to rotate
    t = torch.arange(seq_len).float() ## This is the 256 numbers that we have. 
    updated_frequency = torch.outer(t, roatating_frequency)
    return updated_frequency.cos(), updated_frequency.sin() ## Returns two matrices of size 256 X 4 which represents the sin and cos positions of the token positions 
def apply_rope(x, cos, sin):
    T = x.shape[1] ## Exatracting the T'th element from B,T,C i.e. 256 in this case 
    x1 = x[:, :, 0::2] ## Only tshe even dimensions from C which is 8 in this case
    x2 = x[:, :, 1::2] ## Only the odd dimensions from C which is 8 in this case
    cos, sin = cos[:T], sin[:T] ## Getting the size of matrix based on the T 
    rot1 = x1 *cos -x2 * sin ## Rotating through sin and cos
    rot2 = x1 * sin + x2 * cos ## Rotating through sin and cos 
    return torch.stack([rot1, rot2], dim=-1).flatten(-2) ## Changing the shape form (256,4,2) -> (256,8)


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
    y = torch.stack([data[i+1:i+block_size+1] for i in index]) ## 64 X 256
    return x.long(), y.long() ## 64 X 256 , ##64 X 256

xb, yb = create_batches("train")

## Creating a Head
class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)
        self.query = nn.Linear(n_embd, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        cos, sin = build_rope_cache(block_size, head_size, device)
        self.register_buffer('rope_cos', cos, persistent=False) ## Since they are not the learnable parameters we use the buffer to deal with them
        self.register_buffer('rope_sin', sin, persistent=False)
        self.dropout = nn.Dropout(dropout_layer)
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        k = apply_rope(k, self.rope_cos, self.rope_sin) ## Just rotates the given buffer based on what is inside
        q = apply_rope(q, self.rope_cos, self.rope_sin)
        wei = q @ k.transpose(-2,-1) * self.head_size ** -0.5 ## Weight matrix based on the rotated values
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) ### Filling the buffer 
        wei = F.softmax(wei, dim = -1)
        out = wei @ v 
        return out 

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.sa_head = nn.ModuleList([Head(head_size) for _ in range(n_head)]) 
        self.proj = nn.Linear(head_size * n_head, n_embd)
    def forward(self,x):
        out = torch.cat([h(x) for h in self.sa_head], dim = -1) ## The learned data will be stacked next to each other where each stacked data will have the shape of 256 X 8 creating 256 X 32 dimension matrix which will be used 
        out = self.proj(out) ## 256 X 32 @ 32 X 32 -> 256 X 32 
        return out  ## 256 X 32


## Creating a feed forward layer 
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
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
        self.sa_head = MultiHeadAttention(n_head, head_size) ##4 heads with each's dimensoin of 8. The output will be the shape of 256 X 32 
        self.ffwd = FeedForward(n_embd) ## The output will be the shape of 256 X 32 
        self.ln1 = nn.LayerNorm(n_embd) ## 256 X 32
        self.ln2 = nn.LayerNorm(n_embd) ## 256 X 32 
    def forward(self,x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x ## 256 X 32
    
class zalGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_table = nn.Embedding(vocab_size, n_embd) ## 50000+ X 32
        self.blocks = nn.Sequential(
            Block(4, n_embd),
            Block(4, n_embd),
            Block(4, n_embd)
        ) 
        self.ln_f = nn.LayerNorm(n_embd)
        self.linear_layer = nn.Linear(n_embd, vocab_size)

    def forward(self,x, targets = None):
        B, T = x.shape ## 64 X 256
        x = self.embedding_table(x) ## 64 X 256 X 32
        x = self.blocks(x) ## 64 X 256 X 32
        x = self.ln_f(x) ## 64 X 256 X 32
        logits = self.linear_layer(x) ## 65 X 256 X 50000+ 
        if targets is None: 
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)
        return logits, loss ## Logits and Loss updated
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_condition = idx[:,-block_size:] ## Doesnot matter how many batches, or the size of C, but it needs to be less than 256 token
            logits, loss = self(idx_condition)
            logits = logits[:,-1,:] ## The last token of the given sequence to predict the coming token 
            probs = F.softmax(logits, dim = -1)## In each row it converts raw score to probability 
            idx_next = torch.multinomial(probs, num_samples=1) ## Giving one letter at a time 
            idx = torch.cat((idx, idx_next), dim = 1)
        return idx

model = zalGPT()
model.to(device)
## Creating a loss function 
@torch.no_grad()
def generate_loss():
    out = {}
    model.eval()
    for split in ['train','test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = create_batches(split)
            xb, yb = xb.to(device), yb.to(device)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out ## This returns the average of the 200 batches of losses for both training and validation


## Setting up the optimizer 
optimizer = torch.optim.AdamW(model.parameters(), lr= learning_rate)
start_time = time.perf_counter()

loss_history = []
for iter in range(max_iters):
    if iter % eval_interval == 0:        
        loss = generate_loss()
        loss_history.append({"step":iter, "train":loss['train'].item(), "test": loss['test'].item()})
        print(f"step{iter}: Test loss {loss['test']}, Train loss {loss['train']}")
        with open('./loss_tracker/rope.json', 'w') as f:
            json.dump(loss_history, f, indent=2)
    xb,yb = create_batches('train')
    optimizer.zero_grad(set_to_none=True)
    xb,yb = xb.to(device), yb.to(device)
    logits, loss = model(xb,yb)
    loss.backward()
    optimizer.step()


end_time = time.perf_counter()
print(f"Total time taken: {end_time - start_time}")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
## Starting the engine 
context = torch.zeros((1,1), dtype = torch.long, device = device)
print(enc.decode(model.generate(context,5000)[0].tolist()))



