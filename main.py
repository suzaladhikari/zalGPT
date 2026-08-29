import torch 
import torch.nn as nn 
from torch.nn import functional as F
import tiktoken

torch.manual_seed(455841)
### Extracting the key words
chars = set() 
with open("../localgpt/data/TinyStories-train.txt", "r", encoding="utf-8") as f:
    for each_line in f:
        chars.update(f)

chars = sorted(chars)
print(chars)