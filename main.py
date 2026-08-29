import torch 
import torch.nn as nn 
from torch.nn import functional as F
import tiktoken

torch.manual_seed(455841)
### Extracting the key words 
with open("./data/TinyStories-train.txt", "r") as f:
    text = f.read()

chars = sorted(list(set(text)))