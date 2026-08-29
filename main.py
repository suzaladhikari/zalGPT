import torch 
import torch.nn as nn 
from torch.nn import functional as F
import tiktoken
import json

torch.manual_seed(455841)
### Extracting the key words
chars = set() 
with open("../localgpt/data/TinyStories-train.txt", "r", encoding="utf-8") as f:
    for each_line in f:
        chars.update(f)

chars = sorted(chars)

with open('../localgpt/data/vocab.json', 'w', encoding='utf-8') as f:
    json.dump(chars,f)

print(chars)