"""
Unified Training Script for LLM-200M
Combines pre-training and fine-tuning in a single task with optimal hyperparameters
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoTokenizer
import os
import json
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np

from model import LLM200M
from data_processing import create_dataloaders


class Trainer:
    """Unified trainer for pre-training and fine-tuning"""
    def __init__(
        self,
        model: nn.Module,
        dataloaders: dict,
        device: str = 'cuda',
        output_dir: str = './training_outputs',
        checkpoint_dir: str = './checkpoints',
        stats_dir: str = './training_stats',
        # Pre-training hyperparameters (prevent underfitting)
        pretrain_epochs: int = 3,
        pretrain_lr: float = 3e-4,
        pretrain_warmup_steps: int = 2000,
        # Fine-tuning hyperparameters (prevent overfitting)
        finetune_epochs: int = 2,
        finetune_lr: float = 1e-4,
        finetune_warmup_steps: int = 500,
        # Regularization (prevent overfitting)
        weight_decay: float = 0.01,
        grad_clip: float = 1.0,
        dropout: float = 0.1,
        # Optimization
        batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        # Monitoring
        eval_steps: int = 500,
        save_steps: int = 1000,
        log_steps: int = 100,
        early_stopping_patience: int = 5
    ):
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.device = device
        
        # Directories
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        self.stats_dir = stats_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(stats_dir, exist_ok=True)
        
        # Hyperparameters
        self.pretrain_epochs = pretrain_epochs
        self.pretrain_lr = pretrain_lr
        self.pretrain_warmup_steps = pretrain_warmup_steps
        self.finetune_epochs = finetune_epochs
        self.finetune_lr = finetune_lr
        self.finetune_warmup_steps = finetune_warmup_steps
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Monitoring
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.log_steps = log_steps
        self.early_stopping_patience = early_stopping_patience
        
        # Statistics tracking
        self.stats = {
            'pretrain': {
                'losses': [],
                'perplexities': [],
                'learning_rates': [],
                'timestamps': []
            },
            'finetune': {
                'train_losses': [],
                'val_losses': [],
                'train_perplexities': [],
                'val_perplexities': [],
                'learning_rates': [],
                'timestamps': []
            }
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def create_optimizer_and_scheduler(self, phase: str, total_steps: int):
        """Create optimizer and learning rate scheduler"""
        # Optimizer with weight decay (L2 regularization)
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.pretrain_lr if phase == 'pretrain' else self.finetune_lr,
            betas=(0.9, 0.95),  # Beta2=0.95 for better stability
            eps=1e-8,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler with warmup
        warmup_steps = self.pretrain_warmup_steps if phase == 'pretrain' else self.finetune_warmup_steps
        
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # Cosine annealing after warmup
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-6
        )
        
        # Combine schedulers
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )
        
        return optimizer, scheduler
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        total_batches = 0
        
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            _, loss = self.model(input_ids, labels=labels)
            total_loss += loss.item()
            total_batches += 1
        
        avg_loss = total_loss / total_batches
        perplexity = np.exp(avg_loss)
        
        self.model.train()
        return avg_loss, perplexity
    
    def save_checkpoint(self, epoch: int, phase: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'phase': phase,
            'model_state_dict': self.model.state_dict(),
            'stats': self.stats,
            'best_val_loss': self.best_val_loss
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f'{phase}_latest.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f'{phase}_best.pt')
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best {phase} checkpoint (loss: {self.best_val_loss:.4f})")
    
    def save_statistics(self):
        """Save training statistics to JSON"""
        stats_file = os.path.join(self.stats_dir, 'training_metrics.json')
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"✓ Saved statistics to {stats_file}")
    
    def pretrain(self):
        """Pre-training phase on BookCorpus"""
        print("\n" + "="*80)
        print("PHASE 1: PRE-TRAINING ON BOOKCORPUS")
        print("="*80)
        
        dataloader = self.dataloaders['pretrain']
        total_steps = len(dataloader) * self.pretrain_epochs
        
        optimizer, scheduler = self.create_optimizer_and_scheduler('pretrain', total_steps)
        
        self.model.train()
        global_step = 0
        
        for epoch in range(self.pretrain_epochs):
            print(f"\nEpoch {epoch + 1}/{self.pretrain_epochs}")
            epoch_loss = 0
            
            progress_bar = tqdm(dataloader, desc=f"Pre-training")
            
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                _, loss = self.model(input_ids, labels=labels)
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping (prevent exploding gradients)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                
                epoch_loss += loss.item() * self.gradient_accumulation_steps
                
                # Logging
                if global_step % self.log_steps == 0:
                    current_loss = epoch_loss / (batch_idx + 1)
                    current_ppl = np.exp(current_loss)
                    current_lr = scheduler.get_last_lr()[0]
                    
                    self.stats['pretrain']['losses'].append(current_loss)
                    self.stats['pretrain']['perplexities'].append(current_ppl)
                    self.stats['pretrain']['learning_rates'].append(current_lr)
                    self.stats['pretrain']['timestamps'].append(time.time())
                    
                    progress_bar.set_postfix({
                        'loss': f'{current_loss:.4f}',
                        'ppl': f'{current_ppl:.2f}',
                        'lr': f'{current_lr:.2e}'
                    })
                
                # Save checkpoint
                if global_step % self.save_steps == 0:
                    self.save_checkpoint(epoch, 'pretrain')
                    self.save_statistics()
            
            # End of epoch
            avg_epoch_loss = epoch_loss / len(dataloader)
            avg_epoch_ppl = np.exp(avg_epoch_loss)
            print(f"Epoch {epoch + 1} - Loss: {avg_epoch_loss:.4f}, Perplexity: {avg_epoch_ppl:.2f}")
        
        # Save final pre-training checkpoint
        self.save_checkpoint(self.pretrain_epochs, 'pretrain', is_best=True)
        self.save_statistics()
        
        print("\n✓ Pre-training completed!")
    
    def finetune(self):
        """Fine-tuning phase on OpenCoder"""
        print("\n" + "="*80)
        print("PHASE 2: FINE-TUNING ON OPENCODER")
        print("="*80)
        
        train_loader = self.dataloaders['finetune_train']
        val_loader = self.dataloaders['finetune_val']
        total_steps = len(train_loader) * self.finetune_epochs
        
        optimizer, scheduler = self.create_optimizer_and_scheduler('finetune', total_steps)
        
        self.model.train()
        global_step = 0
        
        for epoch in range(self.finetune_epochs):
            print(f"\nEpoch {epoch + 1}/{self.finetune_epochs}")
            epoch_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Fine-tuning")
            
            for batch_idx, batch in enumerate(progress_bar):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                _, loss = self.model(input_ids, labels=labels)
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                
                epoch_loss += loss.item() * self.gradient_accumulation_steps
                
                # Logging
                if global_step % self.log_steps == 0:
                    current_loss = epoch_loss / (batch_idx + 1)
                    current_ppl = np.exp(current_loss)
                    current_lr = scheduler.get_last_lr()[0]
                    
                    self.stats['finetune']['train_losses'].append(current_loss)
                    self.stats['finetune']['train_perplexities'].append(current_ppl)
                    self.stats['finetune']['learning_rates'].append(current_lr)
                    self.stats['finetune']['timestamps'].append(time.time())
                    
                    progress_bar.set_postfix({
                        'loss': f'{current_loss:.4f}',
                        'ppl': f'{current_ppl:.2f}',
                        'lr': f'{current_lr:.2e}'
                    })
                
                # Evaluation
                if global_step % self.eval_steps == 0:
                    val_loss, val_ppl = self.evaluate(val_loader)
                    self.stats['finetune']['val_losses'].append(val_loss)
                    self.stats['finetune']['val_perplexities'].append(val_ppl)
                    
                    print(f"\nValidation - Loss: {val_loss:.4f}, Perplexity: {val_ppl:.2f}")
                    
                    # Early stopping check
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        self.save_checkpoint(epoch, 'finetune', is_best=True)
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.early_stopping_patience:
                            print(f"\n⚠ Early stopping triggered after {self.patience_counter} evaluations without improvement")
                            self.save_statistics()
                            return
                
                # Save checkpoint
                if global_step % self.save_steps == 0:
                    self.save_checkpoint(epoch, 'finetune')
                    self.save_statistics()
            
            # End of epoch evaluation
            val_loss, val_ppl = self.evaluate(val_loader)
            avg_train_loss = epoch_loss / len(train_loader)
            avg_train_ppl = np.exp(avg_train_loss)
            
            print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Train PPL: {avg_train_ppl:.2f}")
            print(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
            
            # Check for overfitting
            if val_loss > avg_train_loss * 1.5:
                print("⚠ Warning: Model may be overfitting (val_loss >> train_loss)")
        
        # Save final fine-tuning checkpoint
        self.save_checkpoint(self.finetune_epochs, 'finetune', is_best=False)
        self.save_statistics()
        
        print("\n✓ Fine-tuning completed!")
    
    def train(self):
        """Run complete training pipeline: pre-training + fine-tuning"""
        start_time = time.time()
        
        print("\n" + "="*80)
        print("STARTING UNIFIED TRAINING PIPELINE")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Pre-training epochs: {self.pretrain_epochs}")
        print(f"Fine-tuning epochs: {self.finetune_epochs}")
        print(f"Batch size: {self.gradient_accumulation_steps * len(next(iter(self.dataloaders['pretrain']))['input_ids'])}")
        print("="*80)
        
        # Phase 1: Pre-training
        self.pretrain()
        
        # Phase 2: Fine-tuning
        self.finetune()
        
        # Final statistics
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED!")
        print("="*80)
        print(f"Total training time: {hours}h {minutes}m")
        print(f"Final checkpoints saved in: {self.checkpoint_dir}")
        print(f"Training statistics saved in: {self.stats_dir}")
        print("="*80)
        
        # Save final model
        final_model_path = os.path.join(self.output_dir, 'llm_200m_final.pt')
        torch.save(self.model.state_dict(), final_model_path)
        print(f"✓ Final model saved to: {final_model_path}")


def main():
    """Main training function"""
    # Configuration - GTX 1650 Optimized (160M parameters)
    config = {
        'vocab_size': 50257,  # GPT-2 tokenizer vocab size
        'embed_dim': 768,     # Keep high quality
        'num_layers': 16,     # Reduced from 24 to 16 (160M params)
        'num_heads': 12,      # Keep same
        'ff_dim': 3072,       # Keep same
        'max_seq_len': 2048,
        'dropout': 0.1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 1,      # Reduced to 1 for 4GB VRAM (160M model)
        'gradient_accumulation_steps': 4,  # Effective batch = 4
        'pretrain_epochs': 2, # Reduced from 3 to 2
        'finetune_epochs': 2,
        'pretrain_lr': 3e-4,
        'finetune_lr': 1e-4,
        'weight_decay': 0.01,
        'bookcorpus_path': 'archive/BookCorpus3.csv'
    }
    
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Initializing model...")
    model = LLM200M(
        vocab_size=config['vocab_size'],
        embed_dim=config['embed_dim'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout']
    )
    
    print("Creating dataloaders...")
    dataloaders = create_dataloaders(
        tokenizer=tokenizer,
        bookcorpus_path=config['bookcorpus_path'],
        batch_size=config['batch_size']
    )
    
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        dataloaders=dataloaders,
        device=config['device'],
        pretrain_epochs=config['pretrain_epochs'],
        finetune_epochs=config['finetune_epochs'],
        pretrain_lr=config['pretrain_lr'],
        finetune_lr=config['finetune_lr'],
        weight_decay=config['weight_decay'],
        gradient_accumulation_steps=config['gradient_accumulation_steps']
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
