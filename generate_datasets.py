from datasets import load_dataset
import tiktoken
import numpy as np
fw = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train")

tokenizer = tiktoken.get_encoding("gpt2")
def tokenize(text):
    eot = tokenizer._special_tokens["<|endoftext|>"]
    tokens = [eot] + tokenizer.encode_ordinary(text)
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all()
    tokens_np = tokens_np.astype(np.uint16)
    return tokens_np

def write_block(filename, tokens_np):
    with open(filename, "wb") as f:
        f.write(tokens_np.tobytes())

for i in range(len(fw)):
    text = fw[i]["text"]
    tokens_np = tokenize(text)
    write_block(f"fineweb_{i}.bin", tokens_np)
