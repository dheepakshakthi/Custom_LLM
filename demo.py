"""
Quick demo script - run this after training to test the chatbot quickly.
"""
import torch
from chatbot import Chatbot

def quick_demo():
    """Run a quick demo of the chatbot."""
    print("\n" + "="*60)
    print("Cogni-Mamba Chatbot - Quick Demo")
    print("="*60)
    
    # Initialize chatbot
    try:
        bot = Chatbot(model_path='checkpoints/best_model.pt', device='cuda')
    except FileNotFoundError:
        print("\n❌ No trained model found!")
        print("Please train the model first:")
        print("  python train.py")
        return
    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        return
    
    # Test prompts
    test_prompts = [
        "Hello! How are you?",
        "What is machine learning?",
        "Tell me a short story about a robot.",
        "What is Python programming?",
        "Explain neural networks in simple terms.",
    ]
    
    print("\n" + "-"*60)
    print("Running test prompts...")
    print("-"*60 + "\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"Test {i}/{len(test_prompts)}")
        print(f"Prompt: {prompt}")
        print("Bot: ", end="", flush=True)
        
        try:
            response = bot.generate(
                prompt,
                max_new_tokens=100,
                temperature=0.8,
                top_k=50,
                top_p=0.95
            )
            # Clean up response
            response = response.split('\n')[0].strip()
            if not response:
                response = "[No response generated]"
            print(response)
        except Exception as e:
            print(f"[Error: {e}]")
        
        print()
    
    print("-"*60)
    print("Demo complete!")
    print("-"*60)
    print("\nTo start interactive chat, run:")
    print("  python chatbot.py")
    print()


if __name__ == '__main__':
    quick_demo()
