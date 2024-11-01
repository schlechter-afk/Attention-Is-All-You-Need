import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import math
import string
from collections import Counter
import re
import random
import numpy as np
import json
from tqdm import tqdm
import gc

from collections import defaultdict
from typing import List, Tuple, Dict

torch.cuda.empty_cache()
gc.collect()

SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        pe = torch.zeros(seq_len, d_model)

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)

        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False).to(device) # (batch, seq_len, d_model)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.depth = d_model // num_heads

        # Linear layers for queries, keys, and values
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output linear layer
        self.fc_out = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size = query.size(0)

        # Linear transformations
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)

        # Split into heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Scaled dot-product attention
        scaled_attention_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.depth)

        if mask is not None:
            mask = mask.unsqueeze(1)
            # scaled_attention_logits += (mask * -1e9)
            scaled_attention_logits = scaled_attention_logits.masked_fill(mask, -1e9)

        attention_weights = torch.softmax(scaled_attention_logits, dim=-1)

        output = torch.matmul(attention_weights, v)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.d_model)

        return self.fc_out(output)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)

        # Layer normalization and dropout
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, look_ahead_mask: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        # Masked Multi-head Self-Attention
        self_attn_output = self.self_attn(query=x, key=x, value=x, mask=look_ahead_mask)
        x = self.layernorm1(x + self.dropout1(self_attn_output))

        # Cross Multi-head Attention
        cross_attn_output = self.cross_attn(query=x, key=encoder_output, value=encoder_output, mask=padding_mask)
        x = self.layernorm2(x + self.dropout2(cross_attn_output))

        # Feedforward Network
        ffn_output = self.ffn(x)
        x = self.layernorm3(x + self.dropout3(ffn_output))

        return x

class Decoder(nn.Module):
    def __init__(self, target_vocab_size: int, d_model: int, num_heads: int, num_layers: int, d_ff: int, dropout_rate: float, pad_idx: int):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(target_vocab_size, d_model)
#         self.positional_encoding = PositionalEncoding(d_model)
        self.pad_idx = pad_idx
        self.d_model = d_model
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, look_ahead_mask: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        pos_enc = PositionalEncoding(self.d_model, x.size(1))
        x = pos_enc(x)
        x = self.dropout(x)
        
        # Pass through the decoder layers
        for layer in self.layers:
            x = layer(x, enc_output, look_ahead_mask, padding_mask)
        
        return x  # Shape: (batch_size, tgt_seq_len, d_model)

def create_padding_mask(seq_q: torch.Tensor, seq_k: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    Create a padding mask for the attention mechanism.

    Args:
        seq_q (torch.Tensor): Query sequence tensor of shape (batch_size, seq_len_q).
        seq_k (torch.Tensor): Key sequence tensor of shape (batch_size, seq_len_k).
        pad_idx (int): Padding token index.

    Returns:
        torch.Tensor: Padding mask of shape (batch_size, seq_len_q, seq_len_k), where positions
                      with padding tokens in seq_k are True.
    """
    mask = (seq_k == pad_idx).unsqueeze(1)  # Shape: (batch_size, 1, seq_len_k)
    mask = mask.expand(-1, seq_q.size(1), -1)  # Shape: (batch_size, seq_len_q, seq_len_k)
    return mask.to(device)

def create_look_ahead_mask(seq_len: int) -> torch.Tensor:
    """
    Create a look-ahead mask to mask future tokens in the decoder's self-attention.

    Args:
        seq_len (int): Sequence length.

    Returns:
        torch.Tensor: Look-ahead mask of shape (seq_len, seq_len), where positions
                      in the upper triangle (future positions) are True.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask

def create_combined_mask(tgt_input: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    Create a combined mask for the decoder's self-attention.

    Args:
        tgt_input (torch.Tensor): Target input tensor of shape (batch_size, tgt_len).
        pad_idx (int): Padding token index.

    Returns:
        torch.Tensor: Combined mask of shape (batch_size, tgt_len, tgt_len).
    """
    tgt_padding_mask = create_padding_mask(tgt_input, tgt_input, pad_idx)  # Shape: (batch_size, tgt_len, tgt_len)
    look_ahead_mask = create_look_ahead_mask(tgt_input.size(1))            # Shape: (tgt_len, tgt_len)
    look_ahead_mask = look_ahead_mask.unsqueeze(0).expand(tgt_input.size(0), -1, -1)
    combined_mask = tgt_padding_mask | look_ahead_mask
    return combined_mask

