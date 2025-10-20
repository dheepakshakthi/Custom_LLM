"""
Configuration File for LLM-200M
Central place for all hyperparameters and settings
"""

# Model Architecture - GTX 1650 Optimized (160M parameters)
MODEL_CONFIG = {
    'vocab_size': 50257,  # GPT-2 tokenizer vocab size
    'embed_dim': 768,     # Keep high quality
    'num_layers': 16,     # Reduced from 24 to 16 (saves memory)
    'num_heads': 12,      # Keep same
    'ff_dim': 3072,       # Keep same (4x embed_dim)
    'max_seq_len': 2048,
    'dropout': 0.1,
    'tie_weights': True
}

# Training Configuration
TRAINING_CONFIG = {
    # Device
    'device': 'cuda',  # 'cuda' or 'cpu'
    
    # Data
    'bookcorpus_path': 'archive/BookCorpus3.csv',
    'cache_dir': './cache',
    
    # Batch sizes - OPTIMIZED FOR GTX 1650
    'batch_size': 1,       # Reduced to 1 for 4GB VRAM (160M model)
    'gradient_accumulation_steps': 4,  # Effective batch size = 1 * 4 = 4
    'num_workers': 2,      # Reduced for memory
    
    # Pre-training - OPTIMIZED
    'pretrain_epochs': 2,  # Reduced from 3 to 2
    'pretrain_lr': 3e-4,
    'pretrain_warmup_steps': 1000,  # Reduced from 2000
    'pretrain_max_length': 256,     # Reduced from 512 for memory
    
    # Fine-tuning - OPTIMIZED
    'finetune_epochs': 2,
    'finetune_lr': 1e-4,
    'finetune_warmup_steps': 500,
    'finetune_max_length': 512,     # Reduced from 1024 for memory
    
    # Regularization
    'weight_decay': 0.01,
    'grad_clip': 1.0,
    
    # Monitoring - MORE FREQUENT FOR GTX 1650
    'eval_steps': 250,         # More frequent (was 500)
    'save_steps': 500,         # More frequent (was 1000)
    'log_steps': 50,           # More frequent (was 100)
    'early_stopping_patience': 3,  # Reduced from 5
    
    # Directories
    'output_dir': './training_outputs',
    'checkpoint_dir': './checkpoints',
    'stats_dir': './training_stats'
}

# Generation Configuration
GENERATION_CONFIG = {
    'max_new_tokens': 256,
    'temperature': 0.8,
    'top_k': 50,
    'top_p': 0.95,
    'repetition_penalty': 1.1
}

# Tokenizer Configuration
TOKENIZER_CONFIG = {
    'model_name': 'gpt2',  # Base tokenizer to use
    'add_special_tokens': True
}


def get_model_config():
    """Get model configuration"""
    return MODEL_CONFIG.copy()


def get_training_config():
    """Get training configuration"""
    return TRAINING_CONFIG.copy()


def get_generation_config():
    """Get generation configuration"""
    return GENERATION_CONFIG.copy()


def get_tokenizer_config():
    """Get tokenizer configuration"""
    return TOKENIZER_CONFIG.copy()


def print_config():
    """Print all configurations"""
    print("\n" + "="*80)
    print("LLM-200M Configuration".center(80))
    print("="*80)
    
    print("\nModel Configuration:")
    for key, value in MODEL_CONFIG.items():
        print(f"  {key:20s}: {value}")
    
    print("\nTraining Configuration:")
    for key, value in TRAINING_CONFIG.items():
        print(f"  {key:30s}: {value}")
    
    print("\nGeneration Configuration:")
    for key, value in GENERATION_CONFIG.items():
        print(f"  {key:20s}: {value}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    print_config()
