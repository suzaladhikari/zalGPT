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

