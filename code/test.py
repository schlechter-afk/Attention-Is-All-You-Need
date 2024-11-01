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

# Loading the training data for the vocabulary mappings

train_en = load_data("/kaggle/input/translation-task/ted-talks-corpus/train.en")

train_fr = load_data("/kaggle/input/translation-task/ted-talks-corpus/train.fr")

(train_en, train_fr, word2idx_en, idx2word_en,
 word2idx_fr, idx2word_fr, english_lengths, french_lengths) = preprocess_data(train_en, train_fr)


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

max_len_en = 507
max_len_fr = 546

def preprocess_test_data(test_sentences: List[str], word2idx: Dict[str, int], max_len):
    """
    Preprocess test data by adding <sos> and <eos> tokens and padding.
    
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
    
    # Pad sentences to the same max length as training
    test_sentences = pad_sentences(test_sentences, max_len_see)

    # Convert sentences to indices using word2idx mapping
    test_indices = []
    for sentence in test_sentences:
        indices = [word2idx.get(word, word2idx[UNK_TOKEN]) for word in sentence.split()]
        test_indices.append(torch.tensor(indices, dtype=torch.long))

    return test_indices, max_len_see

# Example usage
test_en = load_data("/kaggle/input/translation-task/ted-talks-corpus/test.en")  # Load raw test data
test_en_tensors, max_len_en_test = preprocess_test_data(test_en, word2idx_en, max_len_en)  # Preprocess test data
print("Loaded test english tensors successfully!")

test_fr = load_data("/kaggle/input/translation-task/ted-talks-corpus/test.fr")  # Load raw test data
test_fr_tensors, max_len_fr_test = preprocess_test_data(test_fr, word2idx_fr, max_len_fr)  # Preprocess test data
print("Loaded test french tensors successfully!")

from torch.utils.data import DataLoader, Dataset

class TranslationTestDataset(Dataset):
    def __init__(self, src_tensors, tgt_tensors):
        """
        Dataset for loading test data.

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
test_dataset = TranslationTestDataset(test_en_tensors, test_fr_tensors)

# Create the DataLoader
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)  # Adjust batch_size as needed

src_pad_idx = word2idx_en[PAD_TOKEN]
tgt_pad_idx = word2idx_fr[PAD_TOKEN]

def load_model_from_state_dict(model_class, filename="scratch_transformer_model.pth", device=torch.device('cpu')):
    """
    Load a model's state dictionary from a file.

    Args:
        model_class (nn.Module): The model class to instantiate.
        filename (str): The path to the saved state dictionary.
        device (torch.device): The device to map the model to.

    Returns:
        model (nn.Module): The model with the loaded state dictionary.
    """
    # Initialize the model architecture
    model = model_class().to(device)
    
    # Load the saved state dictionary
    state_dict = torch.load(filename, map_location=device)
    model.load_state_dict(state_dict)
    
    return model

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
)

print(device)
model = load_model_from_state_dict(lambda: model, "/kaggle/input/machine-translation/other/default/1/scratch_transformer_model (1).pth", device)

model.eval()

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

# Example usage
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
test_loss = evaluate_model(model, test_dataloader, criterion, device)
print(f"Average Test Loss: {test_loss:.4f}")

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

def evaluate_test_data(model, test_dataloader, word2idx_fr, idx2word_fr, device):
    """
    Evaluate the Transformer model on the test dataset.
    
    Args:
        model (nn.Module): The trained Transformer model.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        word2idx_fr (Dict[str, int]): Target language vocabulary.
        idx2word_fr (Dict[int, str]): Target language reverse vocabulary.
        device (torch.device): The device to run the model on.
    
    Returns:
        float, float: Average BLEU and ROUGE scores over the test dataset.
    """
    total_bleu = 0.0
    total_rouge = 0.0
    num_sentences = 0

    # Create files to store individual sentence scores
    with open("testbleu.txt", "w") as f:

        batch_bleu_scores = []
        batch_rouge_scores = []
        
        for src_batch, tgt_batch in tqdm(test_dataloader, desc="Evaluating Test Set"):
            curr_bleu_sc = 0.0
            curr_rouge_sc = 0.0
            for i in range(src_batch.size(0)):
                src_sentence = src_batch[i].unsqueeze(0)  # Get single sentence
                reference = tgt_batch[i].tolist()  # Reference (ground truth)

                translation = translate_sentence(model, src_sentence, word2idx_fr, idx2word_fr, max_len=max_len_fr_test + 1, device=device)

                reference_tokens = [idx2word_fr[idx] for idx in reference if idx != word2idx_fr[PAD_TOKEN] and idx != word2idx_fr[SOS_TOKEN] and idx != word2idx_fr[EOS_TOKEN]]
                
                bleu_score = compute_bleu(reference_tokens, translation)
                rouge_score = compute_rouge(reference_tokens, translation)

                total_bleu += bleu_score
                total_rouge += rouge_score
                curr_bleu_sc += bleu_score
                curr_rouge_sc += rouge_score
                num_sentences += 1

                # write the actual reference sentence and the bleu score to a file in format: sentence \t bleu
                reference_sentence = " ".join(reference_tokens)
                f.write(f"{reference_sentence}\t{bleu_score}\n")

            batch_bleu_scores.append(curr_bleu_sc / src_batch.size(0))
            batch_rouge_scores.append(curr_rouge_sc / src_batch.size(0))
    
    batch_bleu_scores.sort()
    batch_rouge_scores.sort()
    trim = int(0.1 * len(batch_bleu_scores))

    # exclude bottom 10% of the batch scores on both sides
    batch_bleu_scores = batch_bleu_scores[trim:]
    batch_rouge_scores = batch_rouge_scores[trim:]

    avg_bleu_trim = sum(batch_bleu_scores) / len(batch_bleu_scores)
    avg_rouge_trim = sum(batch_rouge_scores) / len(batch_rouge_scores)
    
    # Compute average scores
    avg_bleu = total_bleu / num_sentences
    avg_rouge = total_rouge / num_sentences

    return avg_bleu, avg_rouge, avg_bleu_trim, avg_rouge_trim, batch_bleu_scores, batch_rouge_scores


avg_bleu, avg_rouge, avg_bleu_trim, avg_rouge_trim, batch_bleu_scores, batch_rouge_scores = evaluate_test_data(model, test_dataloader, word2idx_fr, idx2word_fr, device)
print(f"Average BLEU Score: {avg_bleu:.4f}")
print(f"Average ROUGE-L Score: {avg_rouge:.4f}")
print(f"Average BLEU Score Trimmed: {avg_bleu_trim:.4f}")
print(f"Average ROUGE-L Score Trimmed: {avg_rouge_trim:.4f}")

import matplotlib.pyplot as plt
from collections import defaultdict

def plot_bleu_vs_sentence_length(bleu_scores_file):
    """
    Plots the average BLEU score as a function of sentence length.

    Args:
        bleu_scores_file (str): Path to the file containing sentences and their BLEU scores.
    
    Returns:
        None
    """
    # Dictionary to store total bleu score and count of sentences by sentence length
    sentence_length_bleu = defaultdict(list)
    
    # Read bleu scores from file
    with open(bleu_scores_file, 'r') as f:
        for line in f:
            sentence, bleu_score = line.split('\t')
            sentence_length = len(sentence.split())
            bleu_score = float(bleu_score.strip())
            sentence_length_bleu[sentence_length].append(bleu_score)
    
    # Calculate average BLEU scores for each sentence length
    lengths = []
    avg_bleu_scores = []
    for length, bleu_list in sorted(sentence_length_bleu.items()):
        avg_bleu = sum(bleu_list) / len(bleu_list)
        lengths.append(length)
        avg_bleu_scores.append(avg_bleu)

    # Plotting the curve
    plt.figure(figsize=(8, 5))
    plt.plot(lengths, avg_bleu_scores, marker='o', linestyle='-', color='blue')
    plt.title('Average BLEU Score vs Sentence Length')
    plt.xlabel('Sentence Length')
    plt.ylabel('Average BLEU Score')
    plt.grid(True)
    plt.show()

# Example usage
bleu_scores_file = 'testbleu.txt'
plot_bleu_vs_sentence_length(bleu_scores_file)