"""
Simple tokenizer utilities for the chatbot.
Uses a basic character-level tokenizer for simplicity, but can be replaced
with a more sophisticated tokenizer like GPT-2 tokenizer from HuggingFace.
"""
import json
import os
from typing import List

class SimpleTokenizer:
    """A simple character-level tokenizer."""
    
    def __init__(self, vocab_file='vocab.json'):
        self.vocab_file = vocab_file
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0
        
        if os.path.exists(vocab_file):
            self.load_vocab()
    
    def build_vocab(self, texts: List[str]):
        """Build vocabulary from a list of texts."""
        chars = set()
        for text in texts:
            chars.update(text)
        
        # Add special tokens
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        vocab_list = special_tokens + sorted(list(chars))
        
        self.char_to_idx = {ch: idx for idx, ch in enumerate(vocab_list)}
        self.idx_to_char = {idx: ch for ch, idx in self.char_to_idx.items()}
        self.vocab_size = len(vocab_list)
        
        # Save vocabulary
        self.save_vocab()
        
        return self.vocab_size
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return [self.char_to_idx.get(ch, self.char_to_idx['<UNK>']) for ch in text]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return ''.join([self.idx_to_char.get(idx, '<UNK>') for idx in token_ids])
    
    def save_vocab(self):
        """Save vocabulary to file."""
        with open(self.vocab_file, 'w', encoding='utf-8') as f:
            json.dump({
                'char_to_idx': self.char_to_idx,
                'idx_to_char': {str(k): v for k, v in self.idx_to_char.items()},
                'vocab_size': self.vocab_size
            }, f, indent=2)
    
    def load_vocab(self):
        """Load vocabulary from file."""
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.char_to_idx = data['char_to_idx']
            self.idx_to_char = {int(k): v for k, v in data['idx_to_char'].items()}
            self.vocab_size = data['vocab_size']


class GPT2Tokenizer:
    """Wrapper for HuggingFace GPT-2 tokenizer (more efficient)."""
    
    def __init__(self):
        try:
            from transformers import GPT2TokenizerFast
            self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.vocab_size = len(self.tokenizer)
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text, add_special_tokens=True)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def __call__(self, text, **kwargs):
        """Make the tokenizer callable like HuggingFace tokenizers."""
        return self.tokenizer(text, **kwargs)
