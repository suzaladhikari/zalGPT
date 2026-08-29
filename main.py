import torch 
import torch.nn as nn 
from torch.nn import functional as F
import tiktoken
import json

torch.manual_seed(455841)
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
data = torch.tensor(enc.encode(text,allowed_special={"<|endoftext|>"}), dtype = torch.long)
print(data)
