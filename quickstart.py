"""
Quick Start Script
Provides easy commands for common tasks
"""

import os
import sys
import subprocess


def print_menu():
    """Print main menu"""
    print("\n" + "="*70)
    print("LLM-200M Quick Start Menu".center(70))
    print("="*70)
    print("\n1. Install dependencies")
    print("2. Start training (pre-train + fine-tune)")
    print("3. Start chatbot (default checkpoint)")
    print("4. Start chatbot (pre-trained checkpoint)")
    print("5. Start chatbot (fine-tuned checkpoint)")
    print("6. Visualize training metrics")
    print("7. Test model architecture")
    print("8. Test data loading")
    print("9. Exit")
    print("\n" + "="*70)


def run_command(command, description):
    """Run a command with description"""
    print(f"\n{description}...")
    print(f"Running: {command}\n")
    try:
        subprocess.run(command, shell=True, check=True)
        print(f"\n✓ {description} completed!")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error: {description} failed!")
        print(f"Error code: {e.returncode}")
    except KeyboardInterrupt:
        print(f"\n⚠ {description} interrupted by user")


def main():
    """Main function"""
    while True:
        print_menu()
        choice = input("Enter your choice (1-9): ").strip()
        
        if choice == '1':
            run_command(
                "pip install -r requirements.txt",
                "Installing dependencies"
            )
        
        elif choice == '2':
            print("\n⚠ Warning: Training can take several hours!")
            confirm = input("Do you want to continue? (y/n): ").strip().lower()
            if confirm == 'y':
                run_command(
                    "python train.py",
                    "Training model (this will take a while)"
                )
        
        elif choice == '3':
            run_command(
                "python chatbot.py",
                "Starting chatbot with default checkpoint"
            )
        
        elif choice == '4':
            run_command(
                "python chatbot.py --checkpoint pretrain",
                "Starting chatbot with pre-trained checkpoint"
            )
        
        elif choice == '5':
            run_command(
                "python chatbot.py --checkpoint finetune",
                "Starting chatbot with fine-tuned checkpoint"
            )
        
        elif choice == '6':
            if os.path.exists('training_stats/training_metrics.json'):
                run_command(
                    "python visualize_metrics.py",
                    "Visualizing training metrics"
                )
            else:
                print("\n✗ No training metrics found!")
                print("Please run training first (option 2)")
        
        elif choice == '7':
            print("\nTesting model architecture...")
            run_command(
                "python -c \"from model import LLM200M; model = LLM200M(); print('Model test passed!')\"",
                "Testing model"
            )
        
        elif choice == '8':
            print("\nTesting data loading (this may take a moment)...")
            print("Note: Will use dummy data if BookCorpus is not available\n")
            run_command(
                "python data_processing.py",
                "Testing data loading"
            )
        
        elif choice == '9':
            print("\nGoodbye! 👋")
            sys.exit(0)
        
        else:
            print("\n✗ Invalid choice! Please enter 1-9")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting... Goodbye! 👋")
        sys.exit(0)
