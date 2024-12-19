import torch
import os
import tiktoken
import numpy as np
from torch.utils.data import Dataset
# use name="sample-10BT" to use the 10BT sample

class TextDataset(Dataset):
    def __init__(self, tokens_per_batch, split="train", batch_size=1024):
        assert split in ["train", "test", "validation"]
        self.tokens_per_batch = tokens_per_batch # number of tokens in a sequence
        self.block_size = tokens_per_batch * batch_size
        self.split = split
        self.dataset = os.path.join('sample-10BT', split)
        self.files = [os.path.join(self.dataset, f"fineweb_{i}.bin") for i in range(len(self.dataset))]
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.batch_size = batch_size
        self.file_token_lengths = self._get_file_token_lengths()
        self.num_batches = sum(self.file_token_lengths) // self.batch_size
        self.current_file_idx = 0
        self.current_file_tokens = None
        self.__open_file(0)
        self.beginning_offset = 0

    def __len__(self):
        return self.num_batches
    
    def __getitem__(self, idx):
        local_idx = idx - self.beginning_offset
        if local_idx + self.tokens_per_batch > len(self.current_file_tokens):
            self.beginning_offset += len(self.current_file_tokens)
            self.__open_file(self.current_file_idx + 1)
        return self.current_file_tokens[local_idx:local_idx + self.tokens_per_batch + 1]
    
    def __open_file(self, idx):
        self.current_file_idx = idx
        self.current_file_tokens = np.frombuffer(open(self.files[idx], "rb").read(self.tokens_per_batch * 2), dtype=np.uint16)

    def collate_fn(self, batch):
        x = torch.tensor(batch)
        y = x[:, 1:].contiguous()
        x = x[:, :-1].contiguous()
        return x, y
    
    def _get_file_token_lengths(self):
        lengths = []
        for file in self.files:
            with open(file, "rb") as f:
                tokens = np.frombuffer(f.read(self.tokens_per_batch * 2), dtype=np.uint16)
                lengths.append(len(tokens))
        return lengths
