import tiktoken
import numpy as np

with open('../localgpt/data/TinyStories-train.txt', 'r', encoding = 'utf-8') as f:
    text = f.read()

enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode(
    text, allowed_special={"<|endoftext|>"}
)
data = np.array(tokens, dtype = np.uint16)
data.tofile('../localgpt/data/train.bin')
print(len(data))