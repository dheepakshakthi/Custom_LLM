"""
Visualize Training Metrics
Plots training statistics from the saved JSON file
"""

import json
import matplotlib.pyplot as plt
import os
from datetime import datetime


def plot_training_metrics(stats_file: str = 'training_stats/training_metrics.json'):
    """
    Create visualizations of training metrics
    
    Args:
        stats_file: path to training_metrics.json
    """
    if not os.path.exists(stats_file):
        print(f"Error: {stats_file} not found!")
        print("Please run training first to generate statistics.")
        return
    
    # Load statistics
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('LLM-200M Training Metrics', fontsize=16, fontweight='bold')
    
    # Pre-training Loss
    if stats['pretrain']['losses']:
        ax = axes[0, 0]
        ax.plot(stats['pretrain']['losses'], color='blue', linewidth=2)
        ax.set_title('Pre-training Loss', fontweight='bold')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
    
    # Pre-training Perplexity
    if stats['pretrain']['perplexities']:
        ax = axes[0, 1]
        ax.plot(stats['pretrain']['perplexities'], color='green', linewidth=2)
        ax.set_title('Pre-training Perplexity', fontweight='bold')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Perplexity')
        ax.grid(True, alpha=0.3)
    
    # Pre-training Learning Rate
    if stats['pretrain']['learning_rates']:
        ax = axes[0, 2]
        ax.plot(stats['pretrain']['learning_rates'], color='orange', linewidth=2)
        ax.set_title('Pre-training Learning Rate', fontweight='bold')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Learning Rate')
        ax.grid(True, alpha=0.3)
    
    # Fine-tuning Train/Val Loss
    if stats['finetune']['train_losses']:
        ax = axes[1, 0]
        ax.plot(stats['finetune']['train_losses'], color='blue', linewidth=2, label='Train')
        if stats['finetune']['val_losses']:
            # Align validation losses with training steps
            val_steps = [i * (len(stats['finetune']['train_losses']) // len(stats['finetune']['val_losses'])) 
                        for i in range(len(stats['finetune']['val_losses']))]
            ax.plot(val_steps, stats['finetune']['val_losses'], color='red', linewidth=2, label='Validation')
        ax.set_title('Fine-tuning Loss', fontweight='bold')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Fine-tuning Train/Val Perplexity
    if stats['finetune']['train_perplexities']:
        ax = axes[1, 1]
        ax.plot(stats['finetune']['train_perplexities'], color='green', linewidth=2, label='Train')
        if stats['finetune']['val_perplexities']:
            val_steps = [i * (len(stats['finetune']['train_perplexities']) // len(stats['finetune']['val_perplexities'])) 
                        for i in range(len(stats['finetune']['val_perplexities']))]
            ax.plot(val_steps, stats['finetune']['val_perplexities'], color='red', linewidth=2, label='Validation')
        ax.set_title('Fine-tuning Perplexity', fontweight='bold')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Perplexity')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Fine-tuning Learning Rate
    if stats['finetune']['learning_rates']:
        ax = axes[1, 2]
        ax.plot(stats['finetune']['learning_rates'], color='orange', linewidth=2)
        ax.set_title('Fine-tuning Learning Rate', fontweight='bold')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Learning Rate')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = 'training_stats/training_plots.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training plots saved to {output_path}")
    
    # Show plot
    plt.show()


def print_training_summary(stats_file: str = 'training_stats/training_metrics.json'):
    """
    Print a summary of training metrics
    
    Args:
        stats_file: path to training_metrics.json
    """
    if not os.path.exists(stats_file):
        print(f"Error: {stats_file} not found!")
        return
    
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    print("\n" + "="*80)
    print("TRAINING SUMMARY".center(80))
    print("="*80)
    
    # Pre-training summary
    if stats['pretrain']['losses']:
        print("\nPRE-TRAINING:")
        print(f"  Initial Loss:       {stats['pretrain']['losses'][0]:.4f}")
        print(f"  Final Loss:         {stats['pretrain']['losses'][-1]:.4f}")
        print(f"  Best Loss:          {min(stats['pretrain']['losses']):.4f}")
        print(f"  Initial Perplexity: {stats['pretrain']['perplexities'][0]:.2f}")
        print(f"  Final Perplexity:   {stats['pretrain']['perplexities'][-1]:.2f}")
        print(f"  Best Perplexity:    {min(stats['pretrain']['perplexities']):.2f}")
        print(f"  Total Steps:        {len(stats['pretrain']['losses']):,}")
    
    # Fine-tuning summary
    if stats['finetune']['train_losses']:
        print("\nFINE-TUNING:")
        print(f"  Initial Train Loss: {stats['finetune']['train_losses'][0]:.4f}")
        print(f"  Final Train Loss:   {stats['finetune']['train_losses'][-1]:.4f}")
        print(f"  Best Train Loss:    {min(stats['finetune']['train_losses']):.4f}")
        
        if stats['finetune']['val_losses']:
            print(f"  Final Val Loss:     {stats['finetune']['val_losses'][-1]:.4f}")
            print(f"  Best Val Loss:      {min(stats['finetune']['val_losses']):.4f}")
            
            # Check for overfitting
            final_train = stats['finetune']['train_losses'][-1]
            final_val = stats['finetune']['val_losses'][-1]
            if final_val > final_train * 1.5:
                print(f"  ⚠ Warning: Possible overfitting detected (val_loss >> train_loss)")
            elif final_val > final_train * 1.2:
                print(f"  ⚠ Slight overfitting detected")
            else:
                print(f"  ✓ Good generalization")
        
        print(f"  Initial Train PPL:  {stats['finetune']['train_perplexities'][0]:.2f}")
        print(f"  Final Train PPL:    {stats['finetune']['train_perplexities'][-1]:.2f}")
        
        if stats['finetune']['val_perplexities']:
            print(f"  Final Val PPL:      {stats['finetune']['val_perplexities'][-1]:.2f}")
        
        print(f"  Total Steps:        {len(stats['finetune']['train_losses']):,}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize training metrics')
    parser.add_argument(
        '--stats-file',
        type=str,
        default='training_stats/training_metrics.json',
        help='Path to training_metrics.json'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Only print summary, no plots'
    )
    
    args = parser.parse_args()
    
    # Print summary
    print_training_summary(args.stats_file)
    
    # Plot metrics
    if not args.no_plot:
        plot_training_metrics(args.stats_file)
