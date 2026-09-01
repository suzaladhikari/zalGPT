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

## Setting up the encoder and decoder 
enc = tiktoken.get_encoding('gpt2')
vocab_size = enc.n_vocab ## This gives the total tokens used in the gpt2 encoder
print(vocab_size)
## Training and validaiton split
data = torch.tensor(np.memmap('../localgpt/data/train.bin', dtype = np.uint16, mode = 'r'))
n = int(0.9 * len(data))
train_data = data[:n] ## Loading the training data
validation_data = data[n:] ## Loading the validation data

## Creating the batches 
def create_batches(split):
    data = train_data if split == 'train' else validation_data
    