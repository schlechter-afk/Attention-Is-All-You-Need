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

def preprocess_sentence(sentence: str) -> str:
    """
    Preprocess a sentence: remove punctuation, non-alphanumeric characters, and extra spaces.
    """
    # Remove punctuation and non-alphanumeric characters
    sentence = re.sub(r"[^a-zA-Z0-9]+", " ", sentence)
    # Strip extra spaces
    sentence = re.sub(r"\s+", " ", sentence).strip()
    # Convert to lower case
    sentence = sentence.lower()
    
    return sentence

def load_data(file_path: str) -> List[str]:
    """
    Load data from a file, one sentence per line.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]

def add_special_tokens(sentences: List[str]) -> List[str]:
    """
    Prepend <sos> and append <eos> token to each sentence.
    """
    return [f"{SOS_TOKEN} {sentence} {EOS_TOKEN}" for sentence in sentences]

def pad_sentences(sentences: List[str], max_len: int) -> List[str]:
    """
    Pad sentences to the maximum length using the <pad> token.
    """
    return [sentence + f" {PAD_TOKEN}" * (max_len - len(sentence.split())) for sentence in sentences]

def build_vocab(sentences: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build word2idx and idx2word mappings from sentences.
    """
    word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1, SOS_TOKEN: 2, EOS_TOKEN: 3}
    idx2word = {0: PAD_TOKEN, 1: UNK_TOKEN, 2: SOS_TOKEN, 3: EOS_TOKEN}
    word_freq = defaultdict(int)

    idx = 4
    for sentence in tqdm(sentences, desc="Building Vocabulary"):
        for word in sentence.split():
            if word not in word2idx:
                word2idx[word] = idx
                idx2word[idx] = word
                idx += 1

    return word2idx, idx2word

def preprocess_data(english_sentences: List[str], french_sentences: List[str]) -> Tuple[List[str], List[str], Dict[str, int], Dict[int, str], Dict[str, int], Dict[int, str]]:
    """
    Full preprocessing pipeline: clean, add tokens, pad, and build vocab.
    """
    # Preprocess sentences
    english_sentences = [preprocess_sentence(s) for s in tqdm(english_sentences, desc="Preprocessing English Sentences")]
    french_sentences = [preprocess_sentence(s) for s in tqdm(french_sentences, desc="Preprocessing French Sentences")]

    # Add <sos> and <eos> tokens
    english_sentences = add_special_tokens(english_sentences)
    french_sentences = add_special_tokens(french_sentences)
    
    # Store original sentence lengths (before padding)
    english_lengths = [len(s.split()) for s in english_sentences]
    french_lengths = [len(s.split()) for s in french_sentences]

    # Pad sentences independently for English and French
    max_len_en = max(english_lengths)  # Max length for English sentences
    max_len_fr = max(french_lengths)   # Max length for French sentences
    
    print(max_len_en)
    print(max_len_fr)

    # Pad sentences
    english_sentences = pad_sentences(english_sentences, max_len_en)
    french_sentences = pad_sentences(french_sentences, max_len_fr)

    # Build vocabularies for English and French
    word2idx_en, idx2word_en = build_vocab(english_sentences)
    word2idx_fr, idx2word_fr = build_vocab(french_sentences)

    return (english_sentences, french_sentences, word2idx_en, idx2word_en, word2idx_fr, idx2word_fr, english_lengths, french_lengths)