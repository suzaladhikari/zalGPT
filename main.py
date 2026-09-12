import torch 
import torch.nn as nn 
from torch.nn import functional as F
import tiktoken
import json
import numpy as np
import time 

torch.manual_seed(455841)

### Setting up the hyperparameters
n_embd = 32
batch_size = 64
block_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dropout_layer = 0.2
eval_iters = 100 ## Each batch loss 
learning_rate = 3e-2
max_iters = 5000
eval_interval = 100
### Each time the model gets the data of total 16,384 tokens/step

### Extracting the key words
chars = set() 
with open("../localgpt/data/TinyStories-train.txt", "r", encoding="utf-8") as f:
    text = f.read()
    chars.update(text)
chars = sorted(chars)

with open('../localgpt/data/vocab.json', 'w', encoding='utf-8') as f:
    json.dump(chars,f)

## Setting up the vocab_size 

## Setting up the encoder and decoder
enc = tiktoken.get_encoding('gpt2')
vocab_size = enc.n_vocab
print(vocab_size)
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
    return x.long(),y.long()

xb,yb = get_batch("train")
xb,yb = xb.to(device), yb.to(device)
print(xb.shape)
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
        return out 
    



## Creating a Multi Head Attention Block
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_head * head_size, n_embd) ## 32 X 32, just to make sure that the learned updated matrix talk to eachother
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim = -1) ## The learned data will be stacked next to each other creating 256 X 32 dimension matrix which will be used 
        out = self.proj(out) ## 256 X 32 @ 32 X 32 -> 256 X 32 
        return out  ## 256 X 32

## Feed Forward 
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd , 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd)
        )
    def forward(self,x):
        return self.net(x)
## Lets create a transformer block 
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) ## 256 X 32
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):
        x = x + self.sa(self.ln1(x)) 
        x = x + self.ffwd(self.ln2(x))
        return x 


## Creating a bigram language model 
class GalGPT(nn.Module):
    def __init__(self):
        ## Creating an embedding table 
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) ## 243 X 32 
        self.position_embedding_table = nn.Embedding(block_size, n_embd) ## 256 X 32
        self.blocks = nn.Sequential(
            Block(n_embd, n_head = 4),
            Block(n_embd, n_head = 4),
            Block(n_embd, n_head = 4)
        )  ## 256 X 32
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    def forward(self, idx, targets = None):
        B, T = idx.shape
        embedding_table = self.token_embedding_table(idx)  ## 64 X 256 X 32 meaning 64 X 256 = 16384 such tokens will have a 32 dimensional representation
        position_table = self.position_embedding_table(torch.arange(T, device = device)) ## 256 X 32 meaning 256 X 32 = 8192 new position vectors will represent each token at the given seqeuence, and since the position is same across each seqeuence it will be applied across batches
        x = embedding_table + position_table ## 64 X 256 X 32, Creating a combined matrix input for self attention blocks that has information of both characters and their positions
        x = self.blocks(x) ## 64 X 256 X 32 
        logits = self.lm_head(x) ## 64 X 256 X 243 ## For each token model produces 243 scores one for each possible token in the vocab.
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits_flat = logits.view(B*T, C) ## 16384, 243
            targets = targets.view(B*T) ##16384
            loss = F.cross_entropy(logits_flat,targets)
        return logits, loss


    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_condition = idx[:,-block_size:] ## For every sequence in the batch, we will get on ly the last 256 tokens.
            logits, loss = self(idx_condition)
            logits = logits[:,-1,:] ## From each sequence we will take the last token and its scores for the possible next token.
            probs = F.softmax(logits, dim = -1) ## Converting the raw scores to probabilty in each row
            idx_next = torch.multinomial(probs, num_samples=1) ## This predicts the next idx based on the probabilty distribution created from the probs
            idx = torch.cat((idx, idx_next), dim = 1)  ## The newly generated token gets added based on the lenght of the max new tokens 
        return idx 



## Intiating the model 
model = GalGPT()
m = model.to(device)

### Creating the loss function to calculate the loss throughout the interval

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            X,Y = X.to(device), Y.to(device)
            logits, loss = model(X,Y)
            losses[k] = loss.item() 
        out[split] = losses.mean()
    model.train()
    return out 
        

## Setting up the optimizer 
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
start_time = time.perf_counter()
loss_history = []
for iter in range(max_iters):
    if iter % eval_interval == 0:        
        loss = estimate_loss()
        loss_history.append({"step":iter, "train":loss['train'].item(), "test": loss['val'].item()})
        print(f"step{iter}: Test loss {loss['val']}, Train loss {loss['train']}")
        with open('./loss_tracker/baseline.json', 'w') as f:
            json.dump(loss_history, f, indent=2)
    xb,yb = get_batch('train')
    optimizer.zero_grad(set_to_none=True)
    xb,yb = xb.to(device), yb.to(device)
    logits, loss = model(xb,yb)
    loss.backward()
    optimizer.step()

end_time = time.perf_counter()
time_taken = end_time - start_time
print(f"Total time taken is: {time_taken}")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
context = torch.zeros((1,1), dtype = torch.long, device = device)
print(enc.decode(m.generate(context, max_new_tokens = 500)[0].tolist()))











