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

from utils import *

train_en = load_data("/kaggle/input/translation-task/ted-talks-corpus/train.en")

train_fr = load_data("/kaggle/input/translation-task/ted-talks-corpus/train.fr")

(train_en, train_fr, word2idx_en, idx2word_en,
 word2idx_fr, idx2word_fr, english_lengths, french_lengths) = preprocess_data(train_en, train_fr)

class TranslationDataset(Dataset):
    def __init__(self, english_sentences, french_sentences, english_lengths, french_lengths, word2idx_en, word2idx_fr):
        self.english_sentences = english_sentences
        self.french_sentences = french_sentences
        self.english_lengths = english_lengths
        self.french_lengths = french_lengths
        self.word2idx_en = word2idx_en
        self.word2idx_fr = word2idx_fr

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        # Tokenize sentences (split into words)
        en_sentence = self.english_sentences[idx].split()
        fr_sentence = self.french_sentences[idx].split()

        # Convert words to indices
        en_indices = [self.word2idx_en.get(word, self.word2idx_en[UNK_TOKEN]) for word in en_sentence]
        fr_indices = [self.word2idx_fr.get(word, self.word2idx_fr[UNK_TOKEN]) for word in fr_sentence]

        # Convert to tensors
        en_tensor = torch.tensor(en_indices, dtype=torch.long)
        fr_tensor = torch.tensor(fr_indices, dtype=torch.long)

        # Get lengths
        en_length = self.english_lengths[idx]
        fr_length = self.french_lengths[idx]

        return en_tensor, fr_tensor, en_length, fr_length

train_dataset = TranslationDataset(train_en, train_fr, english_lengths, french_lengths, word2idx_en, word2idx_fr)

# Parameters
batch_size = 16  # Adjust batch size as needed
shuffle = True   # Shuffle data for training

# Create DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

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

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        
        # Layer normalization and dropout
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Multi-head attention
        attn_output = self.self_attn(x, x, x, mask)
        x = self.layernorm1(x + self.dropout1(attn_output))  # Residual connection and layer norm

        # Feedforward network
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + self.dropout2(ffn_output))  # Residual connection and layer norm

        return x

class Encoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, num_heads: int, num_layers: int, d_ff: int, dropout_rate: float, pad_idx: int):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)

        self.pad_idx = pad_idx  # Padding index for mask creation
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout_rate)

        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, mask) -> torch.Tensor:

        x = self.embedding(x) * math.sqrt(self.d_model)

        pos_enc = PositionalEncoding(self.d_model, x.size(1))
        x = pos_enc(x)

        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return x

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

class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, src_pad_idx: int, tgt_pad_idx: int,
                 num_layers: int = 6, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, dropout_rate: float = 0.1):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, num_layers, d_ff, dropout_rate, src_pad_idx)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, num_layers, d_ff, dropout_rate, tgt_pad_idx)
        self.final_layer = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Transformer model.

        Args:
            src (torch.Tensor): Source input tensor of shape (batch_size, src_len).
            tgt (torch.Tensor): Target input tensor of shape (batch_size, tgt_len).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, tgt_len, tgt_vocab_size).
        """
        # Create masks
        src_mask = create_padding_mask(src, src, src_pad_idx)  # (batch_size, 1, src_len, src_len)
        enc_output = self.encoder(src, src_mask)          # (batch_size, src_len, d_model)

        tgt_input = tgt[:, :-1]   # Exclude the last token for input

        # Create masks for decoder
        self_attn_mask = create_combined_mask(tgt_input, tgt_pad_idx)  # (batch_size, tgt_len, tgt_len)
        cross_attn_mask = create_padding_mask(tgt_input, src, src_pad_idx)  # (batch_size, tgt_len, src_len)

        dec_output = self.decoder(tgt_input, enc_output, self_attn_mask, cross_attn_mask)  # (batch_size, tgt_len, d_model)

        # Final linear layer
        output = self.final_layer(dec_output)  # (batch_size, tgt_len, tgt_vocab_size)

        return output

# Hyperparameters
num_epochs = 10
learning_rate = 1e-4

src_vocab_size = len(word2idx_en)
tgt_vocab_size = len(word2idx_fr)
src_pad_idx = word2idx_en[PAD_TOKEN]
tgt_pad_idx = word2idx_fr[PAD_TOKEN]

d_model = 512
num_layers = 6
num_heads = 8
d_ff = 1024
dropout_rate = 0.1

assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

# # Initialize model
model = Transformer(src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx, num_layers, d_model, num_heads, d_ff, dropout_rate)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

validation_losses = []
train_losses = []

torch.cuda.empty_cache()
gc.collect()

def preprocess_val_data(test_sentences: List[str], word2idx: Dict[str, int], max_len):
    """
    Preprocess val data by adding <sos> and <eos> tokens and padding.
    
    Args:
        test_en_sentences (List[str]): List of raw English test sentences.
        word2idx_en (Dict[str, int]): Source language vocabulary mapping (word2idx).
        max_len_en (int): Maximum sequence length for source language (same as used during training).
    
    Returns:
        List[torch.Tensor]: List of tokenized, padded tensors for test data.
    """
    # Preprocess sentences: lowercase, remove punctuation, etc.
    test_sentences = [preprocess_sentence(s) for s in test_sentences]

    # Add <sos> and <eos> tokens
    test_sentences = add_special_tokens(test_sentences)
    
    # Store original sentence lengths (before padding)
    lengths = [len(s.split()) for s in test_sentences]

    max_len_see = max(lengths) 
    print(f"Max lens see: {max_len_see}")
    
    # Pad sentences to the same max length 
    test_sentences = pad_sentences(test_sentences, max_len_see)

    # Convert sentences to indices using word2idx mapping
    test_indices = []
    for sentence in test_sentences:
        indices = [word2idx.get(word, word2idx[UNK_TOKEN]) for word in sentence.split()]
        test_indices.append(torch.tensor(indices, dtype=torch.long))

    return test_indices, max_len_see

max_len_en = 507
max_len_fr = 546

val_en = load_data("/kaggle/input/translation-task/ted-talks-corpus/dev.en")  # Load raw test data
val_en_tensors, max_len_en_val = preprocess_val_data(val_en, word2idx_en, max_len_en)  # Preprocess test data
print("Loaded test english tensors successfully!")

val_fr = load_data("/kaggle/input/translation-task/ted-talks-corpus/dev.fr")  # Load raw test data
val_fr_tensors, max_len_fr_val = preprocess_val_data(val_fr, word2idx_fr, max_len_fr)  # Preprocess test data
print("Loaded test french tensors successfully!")

class TranslationValDataset(Dataset):
    def __init__(self, src_tensors, tgt_tensors):
        """
        Dataset for loading val data.

        Args:
            src_tensors (List[torch.Tensor]): List of preprocessed source (English) tensors.
            tgt_tensors (List[torch.Tensor]): List of preprocessed target (French) tensors.
        """
        self.src_tensors = src_tensors
        self.tgt_tensors = tgt_tensors

    def __len__(self):
        return len(self.src_tensors)

    def __getitem__(self, idx):
        return self.src_tensors[idx], self.tgt_tensors[idx]

# Create the test dataset
val_dataset = TranslationValDataset(val_en_tensors, val_fr_tensors)

# Create the DataLoader
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)  # Adjust batch_size as needed

def evaluate_model(model, test_dataloader, criterion, device):
    """
    Evaluate the Transformer model on the test dataset.

    Args:
        model (nn.Module): The trained Transformer model.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        criterion (nn.CrossEntropyLoss): Loss function.
        device (torch.device): Device to run the evaluation on.

    Returns:
        float: The average loss over the test dataset.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    progress_bar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc=f"Evaluating...")
    
    with torch.no_grad():
        for batch_idx, batch in progress_bar:
            src_batch, tgt_batch = batch
            # Move data to device
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            # Forward pass
            outputs = model(src_batch, tgt_batch)

            # Prepare target output (shifted by 1 to exclude <sos>)
            tgt_output = tgt_batch[:, 1:].contiguous().view(-1)  # (batch_size * (tgt_len - 1))

            # Reshape model outputs for loss computation
            outputs = outputs.view(-1, outputs.size(-1))  # (batch_size * (tgt_len - 1), tgt_vocab_size)

            # Compute loss
            loss = criterion(outputs, tgt_output)

            # Accumulate total loss
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            num_batches += 1
            

    # Calculate average loss
    avg_loss = total_loss / num_batches
    return avg_loss

torch.cuda.empty_cache()
gc.collect()

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch_idx, batch in progress_bar:
        src_batch, tgt_batch, src_lengths, tgt_lengths = batch

        # Move tensors to device
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)

        # Forward pass
        
        outputs = model(src_batch, tgt_batch)
        outputs = outputs.view(-1, tgt_vocab_size)
        
        tgt_output = tgt_batch[:, 1:]
        tgt_output = tgt_output.contiguous().view(-1)

        loss = criterion(outputs, tgt_output)
        
        torch.cuda.empty_cache()
        gc.collect()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    average_loss = epoch_loss / len(train_dataloader)
    
    train_losses.append(average_loss)
    
    val_loss = evaluate_model(model, val_dataloader, criterion, device)
    
    validation_losses.append(val_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {average_loss:.4f}, Val Loss: {val_loss:.4f}")

def save_model(model, filename="model.pth"):
    """
    Save the model's state dictionary to a file.

    Args:
        model (nn.Module): The model to save.
        filename (str): The path where the model will be saved.
    """
    torch.save(model.state_dict(), filename)
    
save_model(model, "scratch_transformer_model.pth")

import torch
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

def translate_sentence(model, src_seq, word2idx_fr, idx2word_fr, max_len=50, device='cpu'):
    """
    Generate a translation for a single source sentence.
    
    Args:
        model (nn.Module): The trained Transformer model.
        src_seq (torch.Tensor): The source sequence tensor.
        word2idx_fr (Dict[str, int]): Target language vocabulary mapping.
        idx2word_fr (Dict[int, str]): Target language reverse vocabulary mapping.
        max_len (int): Maximum length of the generated translation.
        device (torch.device): The device to run the model on.
    
    Returns:
        List[str]: The generated translation as a list of tokens.
    """
    model.eval()
    src_seq = src_seq.to(device)
    src_mask = create_padding_mask(src_seq, src_seq, word2idx_fr[PAD_TOKEN])

    with torch.no_grad():
        # Encode the source sentence
        enc_output = model.encoder(src_seq, src_mask)

        # Initialize the target sentence with <sos> token
        tgt_indices = [word2idx_fr[SOS_TOKEN]]

        for _ in range(max_len):
            tgt_input = torch.tensor(tgt_indices).unsqueeze(0).to(device)
            
            tgt_mask = create_combined_mask(tgt_input, word2idx_fr[PAD_TOKEN])
            
            cross_attn_mask = create_padding_mask(tgt_input, src_seq, src_pad_idx)            
            
            # Decode step-by-step
            dec_output = model.decoder(tgt_input, enc_output, tgt_mask, cross_attn_mask)
            logits = model.final_layer(dec_output)
            
            # Get the predicted token (take the token with the highest probability)
            next_token_logits = logits[:, -1, :]  # Last token
            next_token = torch.argmax(next_token_logits, dim=-1).item()
            
            if next_token == word2idx_fr[EOS_TOKEN]:
                break

            tgt_indices.append(next_token)

    # Convert token indices to words
    translation_tokens = [idx2word_fr[idx] for idx in tgt_indices[1:]]  # Skip the <sos> token
    return translation_tokens

from nltk.translate.bleu_score import SmoothingFunction

def compute_bleu(reference, translation):
    """
    Compute BLEU score for a single reference and translation.
    
    Args:
        reference (List[str]): The reference sentence as a list of tokens.
        translation (List[str]): The translated sentence as a list of tokens.
    
    Returns:
        float: BLEU score.
    """
    smoothing_function = SmoothingFunction().method7
    if len(translation) == 1:
        # If translation is a single word, compute BLEU-1
        weights = (1.0,)
    else:
        # Use default weights or adjust as needed
        weights = (0.25, 0.25, 0.25, 0.25)
    return sentence_bleu(
        [reference], 
        translation, 
        smoothing_function=smoothing_function, 
        weights=weights
    )

def compute_rouge(reference, translation):
    """
    Compute ROUGE score for a single reference and translation.
    
    Args:
        reference (List[str]): The reference sentence as a list of tokens.
        translation (List[str]): The translated sentence as a list of tokens.
    
    Returns:
        Dict: ROUGE-L score.
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(' '.join(reference), ' '.join(translation))
    return scores['rougeL'].fmeasure


## Code for Hyperparameter tuning:

def evaluate_model_metrics(model, dataloader, word2idx_src, word2idx_tgt, idx2word_tgt, device):
    model.eval()
    src_pad_idx = word2idx_src[PAD_TOKEN]
    tgt_pad_idx = word2idx_tgt[PAD_TOKEN]
    total_bleu = 0.0
    total_rouge = 0.0
    num_sentences = 0
    
    with torch.no_grad():
        
        batch_bleu_scores = []
        batch_rouge_scores = []

        for src_batch, tgt_batch in dataloader:

            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            curr_bleu_sc = 0.0
            curr_rouge_sc = 0.0

            # Generate translations
            for i in range(src_batch.size(0)):
                src_seq = src_batch[i].unsqueeze(0)
                tgt_seq = tgt_batch[i]
                
                translation = translate_sentence(
                    model, src_seq, word2idx_tgt, idx2word_tgt, max_len=max_len_fr_val, device=device
                )
                
                # Reference sentence
                reference_indices = tgt_seq.tolist()
                reference_tokens = [idx2word_tgt[idx] for idx in reference_indices if idx != word2idx_tgt[EOS_TOKEN] and idx != word2idx_tgt[PAD_TOKEN] and idx != word2idx_tgt[SOS_TOKEN]]
                
                # Compute BLEU score
                bleu_score = compute_bleu(reference_tokens, translation)
                total_bleu += bleu_score
                curr_bleu_sc += bleu_score

                # Compute ROUGE-L score
                rouge_score = compute_rouge(reference_tokens, translation)
                total_rouge += rouge_score
                curr_rouge_sc += rouge_score

                num_sentences += 1

            batch_bleu_scores.append(curr_bleu_sc / src_batch.size(0))
            batch_rouge_scores.append(curr_rouge_sc / src_batch.size(0))

    # take the trimmed mean of the batch scores by excluding bottom 10% only
    batch_bleu_scores.sort()
    batch_rouge_scores.sort()
    trim = int(0.1 * len(batch_bleu_scores))

    # exclude bottom 10% of the batch scores only
    batch_bleu_scores = batch_bleu_scores[trim:]
    batch_rouge_scores = batch_rouge_scores[trim:]

    avg_bleu_trim = sum(batch_bleu_scores) / len(batch_bleu_scores)
    avg_rouge_trim = sum(batch_rouge_scores) / len(batch_rouge_scores)

    avg_bleu = total_bleu / num_sentences
    avg_rouge = total_rouge / num_sentences
    
    return avg_bleu, avg_rouge, avg_bleu_trim, avg_rouge_trim

def train_and_evaluate(hyperparams):
    # Unpack hyperparameters
    num_layers = hyperparams['num_layers']
    num_heads = hyperparams['num_heads']
    d_model = hyperparams['d_model']
    d_ff = hyperparams['d_ff']
    dropout_rate = hyperparams['dropout_rate']

    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    
    print(f"\nTraining with hyperparameters: {hyperparams}\n")

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        src_pad_idx=src_pad_idx,
        tgt_pad_idx=tgt_pad_idx,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout_rate=dropout_rate
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    bleu_scores = []
    rouge_scores = []
    bleu_trimmed_scores = []
    rouge_trimmed_scores = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in progress_bar:
#             print(batch)
            src_batch, tgt_batch, _, _ = batch
            
            # Move tensors to device
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(src_batch, tgt_batch)
            outputs = outputs.view(-1, tgt_vocab_size)
            
            tgt_output = tgt_batch[:, 1:]
            tgt_output = tgt_output.contiguous().view(-1)
            
            loss = criterion(outputs, tgt_output)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            torch.cuda.empty_cache()
            gc.collect()
        
        # Calculate average training loss
        avg_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Evaluate model on validation set
        avg_val_loss = evaluate_model(model, val_dataloader, criterion, device)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        # Compute BLEU and ROUGE scores on validation set
        avg_bleu, avg_rouge, avg_bleu_trim, avg_rouge_trim = evaluate_model_metrics(model, val_dataloader, word2idx_en, word2idx_fr, idx2word_fr, device)
        bleu_scores.append(avg_bleu)
        rouge_scores.append(avg_rouge)
        bleu_trimmed_scores.append(avg_bleu_trim)
        rouge_trimmed_scores.append(avg_rouge_trim)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, BLEU: {avg_bleu:.4f}, ROUGE-L: {avg_rouge:.4f}")
    
    # Return collected metrics
    return {
        'hyperparams': hyperparams,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'bleu_scores': bleu_scores,
        'rouge_scores': rouge_scores,
        'bleu_trimmed_scores': bleu_trimmed_scores,
        'rouge_trimmed_scores': rouge_trimmed_scores
    }

hyperparameter_configs = [
    {
        'num_layers': 4,
        'num_heads': 4,
        'd_model': 128,
        'd_ff': 256,
        'dropout_rate': 0.2
    },
    {
        'num_layers': 4,
        'num_heads': 4,
        'd_model': 256,
        'd_ff': 512,
        'dropout_rate': 0.1
    },
    {
        'num_layers': 6,
        'num_heads': 8,
        'd_model': 512,
        'd_ff': 1024,
        'dropout_rate': 0.1
    }
]

num_epochs = 7
learning_rate = 1e-4

results = []
for config in hyperparameter_configs:
    result = train_and_evaluate(config)
    results.append(result)
    
    # Clear cache after each hyperparameter configuration
    torch.cuda.empty_cache()
    gc.collect()

## Code for Hyperparameter tuning complete

## Code for plotting the results:

import matplotlib.pyplot as plt

# Plot train and validation loss curves
plt.figure(figsize=(20, 10))
for result in results:
    epochs = list(range(1, len(result['train_losses']) + 1))
    plt.plot(epochs, result['train_losses'], label=f"Train Loss: {result['hyperparams']}")
    plt.plot(epochs, result['val_losses'], label=f"Val Loss: {result['hyperparams']}", linestyle='--')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Loss Curves')
plt.legend()
plt.grid(True)
plt.show()

# Plot maximum BLEU scores for each hyperparameter setting
max_bleu = [max(result['bleu_scores']) for result in results]
hyperparams = [f"{res['hyperparams']['num_layers']} layers, {res['hyperparams']['num_heads']} heads, {res['hyperparams']['dropout_rate']} dropout, {res['hyperparams']['d_ff']} hidden_dim" for res in results]
plt.figure(figsize=(20, 8))
bars = plt.barh(hyperparams, max_bleu, color='skyblue')

# Annotate each bar with its corresponding value
for bar, score in zip(bars, max_bleu):
    plt.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/3, f'{score:.4f}', 
             va='center', fontsize=10)

plt.xlabel('Maximum BLEU Score')
plt.title('Maximum BLEU Scores Across Epochs for All Hyperparameters')
plt.grid(True, axis='x')
plt.show()

# Plot ROUGE score curves
plt.figure(figsize=(20, 10))
for result in results:
    epochs = list(range(1, len(result['rouge_scores']) + 1))
    plt.plot(epochs, result['rouge_scores'], label=f"ROUGE-L: {result['hyperparams']}")

plt.xlabel('Epoch')
plt.ylabel('ROUGE-L Score')
plt.title('ROUGE-L Score Curves Across Hyperparameters')
plt.legend()
plt.grid(True)
plt.show()

# Plot final train and validation losses for all hyperparameters
final_train_losses = [result['train_losses'][-1] for result in results]
final_val_losses = [result['val_losses'][-1] for result in results]

plt.figure(figsize=(20, 10))
width = 0.35  # Bar width
x = list(range(len(hyperparams)))

plt.bar(x, final_train_losses, width, label='Train Loss', color='blue')
plt.bar([p + width for p in x], final_val_losses, width, label='Validation Loss', color='red')

plt.xlabel('Hyperparameters')
plt.ylabel('Final Loss')
plt.title('Final Train vs Validation Loss Comparison')
plt.xticks([p + width / 2 for p in x], hyperparams)
plt.legend()
plt.grid(True)
plt.show()