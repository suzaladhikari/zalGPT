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
eval_iters = 100 ## Each batch loss 
learning_rate = 3e-3
max_iters = 500
eval_interval = 100

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
    index = torch.randint(len(data)-block_size, (batch_size,)) ## Random integers of the size batch 
    xb = torch.stack([data[i:i+block_size] for i in index]) ## 64 X 256 
    yb = torch.stack([data[i+1: i+block_size+1] for i in index]) ## 64 x 256
    return xb.long(),yb.long()

xb, yb = creating_batches('train')
xb.to(device), yb.to(device)
def build_rope_cache(block_size, head_size, device, theta = 10000):
    rotating_frequency = 1.0 / (theta ** (torch.arange(0,head_size,2).float() / head_size)) ## This defines how much to rotate
    t = torch.arange(block_size).float() ## This defines the posiition that needs to be used 
    updated_frequency = torch.outer(t, rotating_frequency)
    return updated_frequency.cos(), updated_frequency.sin()

def apply_rope(x, cos, sin):
    T = x.shape[1]
    x1 = x[:, :, 0::2] ## Returns the even columns
    x2 = x[:, :, 1::2] ## Returns the odd columns 
    cos, sin = cos[:T], sin[:T] ## Getting the matrix based on T 
    rot1 = x1* cos - x2* sin 
    rot2 = x1 * sin + x2 * cos
    return torch.stack([rot1, rot2] , dim = -1).flatten(-2)


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)
        self.query = nn.Linear(n_embd, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) ## Creates a buffer which wont get modified during the gradient descent or also called mask which basically makes sure that i doesnot see future elements while predicting 
        cos, sin = build_rope_cache(block_size, head_size, device) ## Rotating frequency and storing its value in terms of the sin and cos 
        self.register_buffer('cos', cos, persistent=False) ## 256 X 4
        self.register_buffer('sin', sin, persistent=False) ## 256 X 4 
        self.dropout_layer = nn.Dropout(dropout_layer)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)  ## 32 X 8 
        v = self.value(x) ### 32 X 8 
        q = self.query(x) ## 32 X 8
        k = apply_rope(k, self.cos , self.sin)
        q = apply_rope(q , self.cos, self.sin)
        wei = q @ k.transpose(-2,-1) * self.head_size ** -0.5 ### Applying the self attention 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) ## The masked fill goes before output because if we did after the softmax each row would no longer sum to 1 which will break the probaility distribution 
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        return out 

class MultiHeadAttention(nn.Module):
    def __init__(self, number_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(number_heads)])
        self.proj = nn.Linear(number_heads * head_size , n_embd)
    def forward(self,x):
        x = torch.cat([h(x) for h in self.heads], dim = -1)
        return self.proj(x)
    
        

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
        self.ln1 = nn.RMSNorm(n_embd)
        self.ln2 = nn.RMSNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x 
        
    

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
        self.rm_f = nn.RMSNorm(n_embd)
        self.linear_layer = nn.Linear(n_embd, vocab_size)


    def forward(self,idx, target = None):
        embed = self.embedding_table(idx) ## The shape will be 256 X 32 
        out = self.blocks(embed)
        out = self.rm_f(out)
        logits = self.linear_layer(out)
        if target == None:
            loss = None
        else:
            B,T,C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = target.view(B*T)
            loss = F.cross_entropy(logits_flat,targets_flat)
        return logits, loss 

    def generate(self, idx, max_new_tokens = 5000):
        for _ in range(max_new_tokens):
            idx_limited = idx[:,-block_size:]
            logits, loss = self(idx_limited)
            logits_last = logits[:,-1,:] ## This is only required for us as we predict the upcoming integer 
            distribution = F.softmax(logits_last, dim = -1)
            idx_next = torch.multinomial(distribution, num_samples=1) ## Randomly picking up one index from the given distribution
            idx = torch.cat([idx, idx_next], dim = -1)
        return idx

model = zalGPT()
model.to(device)



## Creating a loss function 
@torch.no_grad()
def generate_loss():
    loss_dict = {}
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)## Torch zeros of batch size 
        for k in range(eval_iters):
            xb,yb = creating_batches(split)
            xb,yb = xb.to(device), yb.to(device)
            logits, loss = model(xb,yb)
            losses[k] = loss.item()
        loss_dict[split] = losses.mean()
    return loss_dict

### Setting up the optimizer 
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = generate_loss()
        print(f"step{iter}: Test loss {losses['test']}, Train loss {losses['train']}")

    xb, yb = creating_batches('train')
    xb,yb = xb.to(device), yb.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits, loss = model(xb,yb) ## For the training purpose only 
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype = torch.long, device = device)
print(enc.decode(model.generate(context,5000)[0].tolist()))




    

        

