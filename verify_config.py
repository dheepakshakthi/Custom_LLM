"""
Configuration Verification Script
Checks if all settings are properly configured for 160M parameter model
"""

import torch
from model import LLM200M
from config import MODEL_CONFIG, TRAINING_CONFIG


def count_parameters(model):
    """Count trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def verify_configuration():
    """Verify all configuration settings"""
    print("\n" + "="*80)
    print("CONFIGURATION VERIFICATION FOR 160M MODEL".center(80))
    print("="*80)
    
    # Check model config
    print("\n1. MODEL ARCHITECTURE:")
    print(f"   Vocabulary Size:   {MODEL_CONFIG['vocab_size']:>10,} (GPT-2 tokenizer)")
    print(f"   Embedding Dim:     {MODEL_CONFIG['embed_dim']:>10,}")
    print(f"   Number of Layers:  {MODEL_CONFIG['num_layers']:>10,} (Target: 16 for 160M)")
    print(f"   Number of Heads:   {MODEL_CONFIG['num_heads']:>10,}")
    print(f"   Feed-Forward Dim:  {MODEL_CONFIG['ff_dim']:>10,}")
    print(f"   Max Sequence Len:  {MODEL_CONFIG['max_seq_len']:>10,}")
    print(f"   Dropout:           {MODEL_CONFIG['dropout']:>10.2f}")
    print(f"   Weight Tying:      {MODEL_CONFIG['tie_weights']!s:>10}")
    
    # Create model and count parameters
    print("\n2. PARAMETER COUNT:")
    model = LLM200M(
        vocab_size=MODEL_CONFIG['vocab_size'],
        embed_dim=MODEL_CONFIG['embed_dim'],
        num_layers=MODEL_CONFIG['num_layers'],
        num_heads=MODEL_CONFIG['num_heads'],
        ff_dim=MODEL_CONFIG['ff_dim'],
        max_seq_len=MODEL_CONFIG['max_seq_len'],
        dropout=MODEL_CONFIG['dropout']
    )
    total_params = count_parameters(model)
    print(f"   Total Parameters:  {total_params:>15,} ({total_params/1e6:.1f}M)")
    
    # Check memory config
    print("\n3. MEMORY OPTIMIZATION (GTX 1650 4GB):")
    print(f"   Batch Size:        {TRAINING_CONFIG['batch_size']:>10,} (Reduced for 4GB VRAM)")
    print(f"   Grad Accumulation: {TRAINING_CONFIG['gradient_accumulation_steps']:>10,}")
    print(f"   Effective Batch:   {TRAINING_CONFIG['batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']:>10,}")
    print(f"   Num Workers:       {TRAINING_CONFIG['num_workers']:>10,}")
    
    # Check training config
    print("\n4. TRAINING SETTINGS:")
    print(f"   Pre-train Epochs:  {TRAINING_CONFIG['pretrain_epochs']:>10,}")
    print(f"   Fine-tune Epochs:  {TRAINING_CONFIG['finetune_epochs']:>10,}")
    print(f"   Pre-train LR:      {TRAINING_CONFIG['pretrain_lr']:>10.6f}")
    print(f"   Fine-tune LR:      {TRAINING_CONFIG['finetune_lr']:>10.6f}")
    print(f"   Weight Decay:      {TRAINING_CONFIG['weight_decay']:>10.4f}")
    print(f"   Gradient Clip:     {TRAINING_CONFIG['grad_clip']:>10.2f}")
    
    # Check sequence lengths
    print("\n5. SEQUENCE LENGTHS:")
    print(f"   Pre-train Max:     {TRAINING_CONFIG['pretrain_max_length']:>10,} (Reduced for memory)")
    print(f"   Fine-tune Max:     {TRAINING_CONFIG['finetune_max_length']:>10,} (Reduced for memory)")
    
    # Check monitoring
    print("\n6. MONITORING:")
    print(f"   Eval Steps:        {TRAINING_CONFIG['eval_steps']:>10,}")
    print(f"   Save Steps:        {TRAINING_CONFIG['save_steps']:>10,}")
    print(f"   Log Steps:         {TRAINING_CONFIG['log_steps']:>10,}")
    print(f"   Early Stop Patience: {TRAINING_CONFIG['early_stopping_patience']:>8,}")
    
    # Check CUDA availability
    print("\n7. HARDWARE:")
    if torch.cuda.is_available():
        print(f"   CUDA Available:    {'Yes':>10}")
        print(f"   GPU Name:          {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory:        {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"   CUDA Available:    {'No':>10} ⚠️ WARNING: Training will be very slow on CPU!")
    
    # Verification summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY".center(80))
    print("="*80)
    
    issues = []
    warnings = []
    
    # Check critical settings
    if MODEL_CONFIG['vocab_size'] != 50257:
        issues.append("❌ Vocab size should be 50257 for GPT-2 tokenizer")
    else:
        print("✓ Vocab size correctly set to 50257")
    
    if MODEL_CONFIG['num_layers'] != 16:
        warnings.append("⚠️  Number of layers is not 16 (expected for 160M)")
    else:
        print("✓ Number of layers set to 16 (160M target)")
    
    if TRAINING_CONFIG['batch_size'] != 1:
        warnings.append("⚠️  Batch size is not 1 (may cause OOM on GTX 1650)")
    else:
        print("✓ Batch size set to 1 (safe for 4GB VRAM)")
    
    if not torch.cuda.is_available():
        warnings.append("⚠️  CUDA not available - training will be very slow")
    else:
        print("✓ CUDA is available")
    
    if total_params < 150e6 or total_params > 200e6:
        warnings.append(f"⚠️  Parameter count ({total_params/1e6:.1f}M) is outside expected range (150M-200M)")
    else:
        print(f"✓ Parameter count ({total_params/1e6:.1f}M) is in acceptable range")
    
    # Print issues/warnings
    if issues:
        print("\n" + "="*80)
        print("CRITICAL ISSUES:")
        for issue in issues:
            print(f"  {issue}")
    
    if warnings:
        print("\n" + "="*80)
        print("WARNINGS:")
        for warning in warnings:
            print(f"  {warning}")
    
    if not issues and not warnings:
        print("\n✅ All settings look good! Ready to start training.")
    
    print("\n" + "="*80)
    
    return len(issues) == 0


if __name__ == "__main__":
    verify_configuration()
