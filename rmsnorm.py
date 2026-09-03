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