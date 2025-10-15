"""
Data loading utilities for training on multiple datasets.
Implements efficient data loading with minimal memory overhead.
Supports: OpenAssistant, Dolly, Alpaca, TinyStories, Code datasets.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Optional
import numpy as np

class TextDataset(Dataset):
    """Efficient dataset for language modeling with multi-dataset support."""
    
    def __init__(self, tokenizer, max_length=512, dataset_name='oasst', split='train', max_samples=None):
        """
        Args:
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            dataset_name: Name of the dataset to load
                         'oasst' - OpenAssistant (recommended)
                         'dolly' - Databricks Dolly-15k
                         'alpaca' - Stanford Alpaca
                         'tinystories' - TinyStories
                         'code_search_net' - Python code
            split: Dataset split ('train', 'validation', 'test')
            max_samples: Limit number of samples for quick testing
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        print(f"Loading {dataset_name} dataset ({split} split)...")
        
        # Load dataset
        try:
            if dataset_name == 'oasst' or dataset_name == 'openassistant':
                self._load_openassistant(split, max_samples)
            elif dataset_name == 'dolly':
                self._load_dolly(split, max_samples)
            elif dataset_name == 'alpaca':
                self._load_alpaca(split, max_samples)
            elif dataset_name == 'tinystories':
                self._load_tinystories(split, max_samples)
            elif dataset_name == 'code_search_net':
                self._load_code_search_net(split, max_samples)
            else:
                # Generic loader
                self._load_generic(dataset_name, split, max_samples)
            
            print(f"Loaded {len(self.examples)} examples")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Using sample data instead...")
            self.examples = self._get_sample_data()
    
    def _load_openassistant(self, split, max_samples):
        """Load OpenAssistant conversational dataset."""
        print("Loading OpenAssistant conversations...")
        dataset = load_dataset('OpenAssistant/oasst1', split='train' if split == 'train' else 'validation')
        
        # Group messages by conversation thread
        conversations = {}
        for item in dataset:
            message_id = item.get('message_id', '')
            parent_id = item.get('parent_id', None)
            text = item.get('text', '')
            role = item.get('role', 'assistant')
            
            if not parent_id:
                # Root message (user question)
                conversations[message_id] = {'user': text, 'assistant': None}
            else:
                # Response to a message
                if parent_id in conversations and role == 'assistant':
                    conversations[parent_id]['assistant'] = text
        
        # Format conversations
        count = 0
        for conv_id, conv in conversations.items():
            if max_samples and count >= max_samples:
                break
            
            if conv['assistant']:
                # Format as instruction-response
                formatted = f"User: {conv['user']}\nAssistant: {conv['assistant']}"
                self.examples.append(formatted)
                count += 1
    
    def _load_dolly(self, split, max_samples):
        """Load Databricks Dolly-15k dataset."""
        print("Loading Dolly-15k...")
        dataset = load_dataset('databricks/databricks-dolly-15k', split='train')
        
        count = 0
        for item in dataset:
            if max_samples and count >= max_samples:
                break
            
            instruction = item.get('instruction', '')
            context = item.get('context', '')
            response = item.get('response', '')
            
            if context:
                text = f"### Instruction: {instruction}\n### Context: {context}\n### Response: {response}"
            else:
                text = f"### Instruction: {instruction}\n### Response: {response}"
            
            if len(text.strip()) > 10:
                self.examples.append(text)
                count += 1
    
    def _load_alpaca(self, split, max_samples):
        """Load Stanford Alpaca dataset."""
        print("Loading Alpaca...")
        dataset = load_dataset('tatsu-lab/alpaca', split='train')
        
        count = 0
        for item in dataset:
            if max_samples and count >= max_samples:
                break
            
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')
            output = item.get('output', '')
            
            if input_text:
                text = f"### Instruction: {instruction}\n### Input: {input_text}\n### Response: {output}"
            else:
                text = f"### Instruction: {instruction}\n### Response: {output}"
            
            if len(text.strip()) > 10:
                self.examples.append(text)
                count += 1
    
    def _load_tinystories(self, split, max_samples):
        """Load TinyStories dataset."""
        print("Loading TinyStories...")
        dataset = load_dataset('roneneldan/TinyStories', split=split)
        
        count = 0
        for item in dataset:
            if max_samples and count >= max_samples:
                break
            
            text = item.get('text', '')
            if text and len(text.strip()) > 10:
                self.examples.append(text)
                count += 1
    
    def _load_code_search_net(self, split, max_samples):
        """Load Code Search Net Python dataset."""
        print("Loading Code Search Net...")
        dataset = load_dataset('code_search_net', 'python', split=split)
        
        count = 0
        for item in dataset:
            if max_samples and count >= max_samples:
                break
            
            code = item.get('func_code_string', '')
            docstring = item.get('func_documentation_string', '')
            
            if docstring:
                text = f"# {docstring}\n{code}"
            else:
                text = code
            
            if text and len(text.strip()) > 10:
                self.examples.append(text)
                count += 1
    
    def _load_generic(self, dataset_name, split, max_samples):
        """Generic loader for other datasets."""
        print(f"Loading {dataset_name}...")
        dataset = load_dataset(dataset_name, split=split)
        
        count = 0
        for item in dataset:
            if max_samples and count >= max_samples:
                break
            
            # Try common text keys
            text = None
            for key in ['text', 'content', 'response', 'output', 'code']:
                if key in item:
                    text = item[key]
                    break
            
            if text and len(text.strip()) > 10:
                self.examples.append(text)
                count += 1
    
    def _get_sample_data(self):
        """Generate sample conversational data for testing."""
        return [
            "User: Hello! How are you today?\nAssistant: I'm doing great, thanks for asking! How can I help you today?",
            "User: What's your favorite programming language?\nAssistant: I love Python because it's versatile and easy to learn. It has great libraries for AI and data science.",
            "User: Can you explain what machine learning is?\nAssistant: Machine learning is a subset of AI that enables systems to learn from data without being explicitly programmed. It uses algorithms to find patterns in data.",
            "User: What are neural networks?\nAssistant: Neural networks are computing systems inspired by biological neural networks that make up animal brains. They consist of layers of interconnected nodes that process information.",
            "User: How do I start learning AI?\nAssistant: Start with Python basics, then move to machine learning libraries like PyTorch or TensorFlow. Practice with datasets on Kaggle and take online courses.",
            "### Instruction: Write a Python function to calculate factorial\n### Response: def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)",
            "User: What is the capital of France?\nAssistant: The capital of France is Paris. It's known for the Eiffel Tower, Louvre Museum, and rich cultural heritage.",
            "User: Explain recursion in simple terms\nAssistant: Recursion is when a function calls itself to solve smaller versions of the same problem. Like Russian nesting dolls - each one contains a smaller version until you reach the smallest one.",
        ] * 200  # Repeat for more training data
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        text = self.examples[idx]
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        
        # Truncate or pad to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Create input and target (shifted by 1 for next token prediction)
        input_ids = tokens[:-1] if len(tokens) > 1 else tokens
        target_ids = tokens[1:] if len(tokens) > 1 else tokens
        
        # Pad sequences
        input_ids = input_ids + [0] * (self.max_length - 1 - len(input_ids))
        target_ids = target_ids + [0] * (self.max_length - 1 - len(target_ids))
        
        return {
            'input_ids': torch.tensor(input_ids[:self.max_length-1], dtype=torch.long),
            'labels': torch.tensor(target_ids[:self.max_length-1], dtype=torch.long),
        }


def create_dataloaders(tokenizer, config, dataset_name='oasst', max_train_samples=50000, max_val_samples=5000):
    """
    Create train and validation dataloaders.
    
    Supported datasets:
    - 'oasst' or 'openassistant': OpenAssistant conversations (RECOMMENDED)
    - 'dolly': Databricks Dolly-15k
    - 'alpaca': Stanford Alpaca
    - 'tinystories': TinyStories
    - 'code_search_net': Python code
    """
    
    # Datasets that don't have a validation split
    needs_manual_split = ['dolly', 'alpaca']
    
    if dataset_name in needs_manual_split:
        # Load full dataset and split manually
        print("Loading dataset for manual train/val split...")
        full_dataset = TextDataset(
            tokenizer=tokenizer,
            max_length=config.max_seq_len,
            dataset_name=dataset_name,
            split='train',
            max_samples=max_train_samples + max_val_samples
        )
        
        # Split 90% train, 10% val
        total_size = len(full_dataset.examples)
        train_size = int(0.9 * total_size)
        
        train_examples = full_dataset.examples[:train_size]
        val_examples = full_dataset.examples[train_size:]
        
        # Create train dataset
        train_dataset = full_dataset
        train_dataset.examples = train_examples
        
        # Create val dataset
        val_dataset = TextDataset.__new__(TextDataset)
        val_dataset.tokenizer = tokenizer
        val_dataset.max_length = config.max_seq_len
        val_dataset.examples = val_examples
    else:
        # Use native train/val splits
        train_dataset = TextDataset(
            tokenizer=tokenizer,
            max_length=config.max_seq_len,
            dataset_name=dataset_name,
            split='train',
            max_samples=max_train_samples
        )
        
        val_dataset = TextDataset(
            tokenizer=tokenizer,
            max_length=config.max_seq_len,
            dataset_name=dataset_name,
            split='validation' if dataset_name in ['oasst', 'openassistant', 'tinystories'] else 'test',
            max_samples=max_val_samples
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader
