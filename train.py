"""
Efficient training script for the LLM with minimal resource usage.
Implements:
- Mixed precision training (FP16)
- Gradient accumulation
- Gradient checkpointing
- Efficient memory management
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import json
from pathlib import Path

from LLM_architecture_168 import CogniMamba, ModelArgs, load_config
from tokenizer_utils import GPT2Tokenizer, SimpleTokenizer
from data_loader import create_dataloaders

class Trainer:
    """Memory-efficient trainer for the LLM."""
    
    def __init__(self, model, config, tokenizer, device='cuda'):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = self._get_lr_scheduler()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.mixed_precision else None #GradScaler is deprecated
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Create checkpoint directory
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def _get_lr_scheduler(self):
        """Create learning rate scheduler with linear warmup and cosine decay."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return float(step) / float(max(1, self.config.warmup_steps))
            return 0.5 * (1.0 + torch.cos(torch.tensor(
                (step - self.config.warmup_steps) / (self.config.max_epochs * 1000 - self.config.warmup_steps) * 3.14159
            )))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Mixed precision forward pass
            if self.scaler:
                with autocast():
                    outputs = self.model(input_ids)
                    # Reshape for loss calculation
                    loss = self.criterion(
                        outputs.view(-1, self.config.vocab_size),
                        labels.view(-1)
                    )
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(input_ids)
                loss = self.criterion(
                    outputs.view(-1, self.config.vocab_size),
                    labels.view(-1)
                )
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
            
            # Update weights after accumulation steps
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1
            
            # Accumulate loss (unscaled)
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item() * self.config.gradient_accumulation_steps:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Periodic evaluation and checkpointing
            if self.global_step % self.config.eval_every == 0:
                torch.cuda.empty_cache()  # Free up memory
            
            if self.global_step % self.config.save_checkpoint_every == 0:
                self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self, val_loader):
        """Evaluate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            if self.scaler:
                with autocast():
                    outputs = self.model(input_ids)
                    loss = self.criterion(
                        outputs.view(-1, self.config.vocab_size),
                        labels.view(-1)
                    )
            else:
                outputs = self.model(input_ids)
                loss = self.criterion(
                    outputs.view(-1, self.config.vocab_size),
                    labels.view(-1)
                )
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Checkpoint loaded: {checkpoint_path}")
    
    def train(self, train_loader, val_loader):
        """Main training loop."""
        print("\n" + "="*50)
        print("Starting Training")
        print("="*50)
        print(f"Total Parameters: {self.model.get_num_params() / 1e6:.2f}M")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.config.mixed_precision}")
        print(f"Batch Size: {self.config.batch_size}")
        print(f"Gradient Accumulation Steps: {self.config.gradient_accumulation_steps}")
        print(f"Effective Batch Size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print("="*50 + "\n")
        
        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Evaluate
            val_loss = self.evaluate(val_loader)
            
            print(f"\nEpoch {epoch + 1}/{self.config.max_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}\n")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pt')
                print(f"New best model saved! Val Loss: {val_loss:.4f}\n")
            
            # Save epoch checkpoint
            self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pt')
        
        print("\n" + "="*50)
        print("Training Complete!")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print("="*50)


def main():
    """Main training function."""
    # Load configuration
    config_dict = load_config('config.json')
    config = ModelArgs(**config_dict)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    try:
        tokenizer = GPT2Tokenizer()
        print(f"Using GPT2Tokenizer, vocab size: {tokenizer.vocab_size}")
        # Update config with actual vocab size
        config.vocab_size = tokenizer.vocab_size
    except ImportError:
        print("GPT2Tokenizer not available, using SimpleTokenizer")
        tokenizer = SimpleTokenizer()
        # Build vocab from dataset later
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        tokenizer=tokenizer,
        config=config,
        dataset_name='oasst',  # OpenAssistant conversations (best for chatbot)
        # Other options: 'dolly', 'alpaca', 'tinystories', 'code_search_net'
        max_train_samples=50000,  # OpenAssistant has good quality data
        max_val_samples=5000
    )
    
    # Update vocab size if using SimpleTokenizer
    if isinstance(tokenizer, SimpleTokenizer) and tokenizer.vocab_size > 0:
        config.vocab_size = tokenizer.vocab_size
    
    # Initialize model
    print("Initializing model...")
    model = CogniMamba(config)
    print(f"Model initialized with {model.get_num_params() / 1e6:.2f}M parameters")
    
    # Initialize trainer
    trainer = Trainer(model, config, tokenizer, device=device)
    
    # Train
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
