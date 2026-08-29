import torch 
import torch.nn as nn 
from torch.nn import functional as F

## Reading the data 
with open("./data/TinyStories-train.txt", "r") as f:
    text = f.read()

print(text[:1000])