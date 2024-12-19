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
    
    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits # (B, T, vocab_size)
        
class Block(nn.Module):
    """
    Transformer Block
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # residual connection
        x = x + self.mlp(self.ln_2(x)) # residual connection
        return x

class MLP(nn.Module):
    """
    Multi-Layer Perceptron
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.config = config

        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x) # gelu used to curb vanishing gradients from relu
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    """
    Causal Self-Attention
    """
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # * 3 for k, v, q
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3)

        # projection output
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # causal mask to ensure that attention is only applied to the past tokens
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, 
            config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, value
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # reshape q, k, v for multi-head attention
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        masked_attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        masked_attn_scores = masked_attn_scores.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        attn = F.softmax(masked_attn_scores, dim=-1)
        y = attn @ v
        #y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y


