from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


@dataclass
class GPT2Config:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT2(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(cls):
        from transformers import GPT2LMHeadModel
        print(f"loading pretrained weights for model gpt2")
        config = GPT2Config()
        model = GPT2(config)
        state_dict = model.state_dict()
        keys = state_dict.keys()
        keys = [key for key in keys if not key.endswith(".attn.bias")]


        model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
        state_dict_hf = model_hf.state_dict()
        keys_hf = state_dict_hf.keys()
        keys_hf = [key for key in keys_hf if not key.endswith(".attn.bias")]
        keys_hf = [key for key in keys_hf if not key.endswith(".attn.masked_bias")]
        transposed_keys_hf = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        assert len(keys) == len(keys_hf), f"mismatched keys: {len(keys)} != {len(keys_hf)}"

        for key in keys_hf:
            if any(key.endswith(transposed_key) for transposed_key in transposed_keys_hf):
                assert state_dict_hf[key].shape[::-1] == state_dict[key].shape
                with torch.no_grad():
                    state_dict[key].copy_(state_dict_hf[key].T)
            else:
                assert state_dict_hf[key].shape == state_dict[key].shape
                with torch.no_grad():
                    state_dict[key].copy_(state_dict_hf[key])
        return model