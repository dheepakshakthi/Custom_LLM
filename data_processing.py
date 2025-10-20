"""
Data Processing Pipeline
Handles both pre-training (BookCorpus) and fine-tuning (OpenCoder) datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
import os
from typing import Dict, List, Optional
import json


class BookCorpusDataset(Dataset):
    """Dataset for pre-training on BookCorpus"""
    def __init__(
        self,
        csv_path: str,
        tokenizer,
        max_length: int = 512,
        cache_dir: str = "./cache"
    ):
        """
        Args:
            csv_path: path to BookCorpus3.csv
            tokenizer: tokenizer instance
            max_length: maximum sequence length
            cache_dir: directory to cache processed data
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Loading BookCorpus from {csv_path}...")
        
        # Load CSV in chunks to handle large files
        try:
            # Try to load cached tokenized data
            cache_file = os.path.join(cache_dir, "bookcorpus_tokenized.pt")
            if os.path.exists(cache_file):
                print(f"Loading cached data from {cache_file}...")
                self.data = torch.load(cache_file)
            else:
                # Read CSV and tokenize - LIMIT to first 20k samples for GTX 1650
                chunks = []
                chunk_size = 5000
                max_samples = 20000  # Limit total samples (reduced from 50k)
                
                for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
                    # Assuming the CSV has a 'text' column
                    if 'text' in chunk.columns:
                        texts = chunk['text'].dropna().tolist()
                    else:
                        # If no 'text' column, use the first column
                        texts = chunk.iloc[:, 0].dropna().tolist()
                    
                    chunks.extend(texts)
                    
                    if len(chunks) % 10000 == 0:
                        print(f"Loaded {len(chunks):,} texts...")
                    
                    # Stop if we've reached max samples
                    if len(chunks) >= max_samples:
                        chunks = chunks[:max_samples]
                        break
                
                self.texts = chunks
                print(f"Total texts loaded: {len(self.texts):,}")
                
                # Tokenize and cache
                print("Tokenizing data (this may take a while)...")
                self.data = []
                batch_size = 1000  # Batch size for tokenization
                
                for i in range(0, len(self.texts), batch_size):
                    batch = self.texts[i:i+batch_size]
                    encodings = self.tokenizer(
                        batch,
                        max_length=max_length,
                        truncation=True,
                        padding='max_length',
                        return_tensors='pt'
                    )
                    self.data.append(encodings['input_ids'])
                    
                    if (i + batch_size) % 5000 == 0:
                        print(f"Tokenized {i + batch_size:,}/{len(self.texts):,} texts...")
                
                self.data = torch.cat(self.data, dim=0)
                
                # Cache for future use
                os.makedirs(cache_dir, exist_ok=True)
                print(f"Caching tokenized data to {cache_file}...")
                torch.save(self.data, cache_file)
                
        except Exception as e:
            print(f"Error loading BookCorpus: {e}")
            print("Creating a small dummy dataset for testing...")
            # Fallback to dummy data
            dummy_texts = [
                "This is a sample text for pre-training.",
                "Language models learn from large amounts of text data.",
                "Deep learning has revolutionized natural language processing."
            ] * 100
            self.data = self.tokenizer(
                dummy_texts,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )['input_ids']
        
        print(f"Dataset ready with {len(self.data):,} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_ids = self.data[idx]
        return {
            'input_ids': input_ids,
            'labels': input_ids.clone()  # For causal LM, labels = inputs
        }


class OpenCoderDataset(Dataset):
    """Dataset for fine-tuning on OpenCoder instruction data"""
    def __init__(
        self,
        tokenizer,
        max_length: int = 1024,
        split: str = 'train',
        max_samples: Optional[int] = None
    ):
        """
        Args:
            tokenizer: tokenizer instance
            max_length: maximum sequence length
            split: 'train' or 'validation'
            max_samples: limit number of samples (for testing)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"Loading OpenCoder dataset (split: {split})...")
        
        try:
            # Load from HuggingFace - use 'educational_instruct' config
            dataset = load_dataset(
                "OpenCoder-LLM/opc-sft-stage2",
                "educational_instruct",  # Specify config
                split=split
            )
            
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            
            self.data = dataset
            print(f"Loaded {len(self.data):,} examples")
            
        except Exception as e:
            print(f"Error loading OpenCoder dataset: {e}")
            print("Creating dummy instruction dataset...")
            # Fallback to dummy data
            self.data = [
                {
                    'instruction': 'Write a Python function to calculate factorial.',
                    'response': 'def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)'
                },
                {
                    'instruction': 'Explain what is recursion.',
                    'response': 'Recursion is a programming technique where a function calls itself to solve a problem.'
                }
            ] * 50
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format as instruction-response pair
        if isinstance(item, dict):
            # Handle different possible formats
            if 'instruction' in item and 'response' in item:
                text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['response']}"
            elif 'prompt' in item and 'completion' in item:
                text = f"### Instruction:\n{item['prompt']}\n\n### Response:\n{item['completion']}"
            elif 'input' in item and 'output' in item:
                text = f"### Instruction:\n{item['input']}\n\n### Response:\n{item['output']}"
            else:
                # Try to use the first two fields
                keys = list(item.keys())
                text = f"### Instruction:\n{item[keys[0]]}\n\n### Response:\n{item[keys[1]]}"
        else:
            text = str(item)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'labels': input_ids.clone()
        }


def create_dataloaders(
    tokenizer,
    bookcorpus_path: str,
    batch_size: int = 8,
    pretrain_max_length: int = 512,
    finetune_max_length: int = 1024,
    num_workers: int = 4,
    cache_dir: str = "./cache"
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for both pre-training and fine-tuning
    
    Args:
        tokenizer: tokenizer instance
        bookcorpus_path: path to BookCorpus CSV
        batch_size: batch size for training
        pretrain_max_length: max length for pre-training
        finetune_max_length: max length for fine-tuning
        num_workers: number of workers for data loading
        cache_dir: cache directory
        
    Returns:
        dict with 'pretrain', 'finetune_train', 'finetune_val' dataloaders
    """
    print("="*80)
    print("Creating DataLoaders")
    print("="*80)
    
    # Pre-training dataset
    pretrain_dataset = BookCorpusDataset(
        csv_path=bookcorpus_path,
        tokenizer=tokenizer,
        max_length=pretrain_max_length,
        cache_dir=cache_dir
    )
    
    pretrain_loader = DataLoader(
        pretrain_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Fine-tuning datasets
    finetune_train_dataset = OpenCoderDataset(
        tokenizer=tokenizer,
        max_length=finetune_max_length,
        split='train'
    )
    
    finetune_train_loader = DataLoader(
        finetune_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Validation set (if available)
    try:
        finetune_val_dataset = OpenCoderDataset(
            tokenizer=tokenizer,
            max_length=finetune_max_length,
            split='validation'
        )
        finetune_val_loader = DataLoader(
            finetune_val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    except:
        # Use a subset of train data as validation
        print("No validation split found, using 5% of training data...")
        val_size = len(finetune_train_dataset) // 20
        train_size = len(finetune_train_dataset) - val_size
        
        finetune_train_subset, finetune_val_subset = torch.utils.data.random_split(
            finetune_train_dataset,
            [train_size, val_size]
        )
        
        finetune_train_loader = DataLoader(
            finetune_train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        finetune_val_loader = DataLoader(
            finetune_val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    print("="*80)
    print(f"Pre-training batches: {len(pretrain_loader):,}")
    print(f"Fine-tuning train batches: {len(finetune_train_loader):,}")
    print(f"Fine-tuning val batches: {len(finetune_val_loader):,}")
    print("="*80)
    
    return {
        'pretrain': pretrain_loader,
        'finetune_train': finetune_train_loader,
        'finetune_val': finetune_val_loader
    }


if __name__ == "__main__":
    # Test data loading
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    dataloaders = create_dataloaders(
        tokenizer=tokenizer,
        bookcorpus_path="archive/BookCorpus3.csv",
        batch_size=4
    )
    
    # Test pre-training data
    batch = next(iter(dataloaders['pretrain']))
    print(f"Pre-train batch keys: {batch.keys()}")
    print(f"Input shape: {batch['input_ids'].shape}")
    
    # Test fine-tuning data
    batch = next(iter(dataloaders['finetune_train']))
    print(f"Fine-tune batch keys: {batch.keys()}")
    print(f"Input shape: {batch['input_ids'].shape}")
