"""
Simple CLI Chatbot using the trained LLM.
Provides an interactive interface for conversing with the model.
"""
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

from LLM_architecture_168 import CogniMamba, ModelArgs
from tokenizer_utils import GPT2Tokenizer, SimpleTokenizer

class Chatbot:
    """CLI Chatbot interface."""
    
    def __init__(self, model_path='checkpoints/best_model.pt', device='cuda'):
        """
        Initialize the chatbot.
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load checkpoint
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model
        config = ModelArgs(**checkpoint['config'])
        self.model = CogniMamba(config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully! ({self.model.get_num_params() / 1e6:.2f}M parameters)")
        
        # Initialize tokenizer
        try:
            self.tokenizer = GPT2Tokenizer()
            print("Using GPT2Tokenizer")
        except ImportError:
            self.tokenizer = SimpleTokenizer()
            print("Using SimpleTokenizer")
        
        self.max_length = config.max_seq_len
        self.conversation_history = []
    
    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=100, temperature=0.8, top_k=50, top_p=0.95):
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated text
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        # Generate tokens
        generated = input_ids
        
        for _ in range(max_new_tokens):
            # Get model predictions
            if generated.shape[1] > self.max_length:
                # Truncate if too long
                input_chunk = generated[:, -self.max_length:]
            else:
                input_chunk = generated
            
            outputs = self.model(input_chunk)
            
            # Get logits for the last token
            logits = outputs[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for end of sequence (if using special tokens)
            if hasattr(self.tokenizer, 'tokenizer') and next_token.item() == self.tokenizer.tokenizer.eos_token_id:
                break
        
        # Decode generated tokens
        generated_ids = generated[0].tolist()
        generated_text = self.tokenizer.decode(generated_ids)
        
        # Extract only the newly generated part
        prompt_decoded = self.tokenizer.decode(input_ids[0].tolist())
        if generated_text.startswith(prompt_decoded):
            generated_text = generated_text[len(prompt_decoded):]
        
        return generated_text.strip()
    
    def chat(self):
        """Start an interactive chat session."""
        print("\n" + "="*60)
        print("🤖 Cogni-Mamba Chatbot")
        print("="*60)
        print("Type your message and press Enter to chat.")
        print("Commands:")
        print("  - 'quit' or 'exit': Exit the chatbot")
        print("  - 'clear': Clear conversation history")
        print("  - 'history': Show conversation history")
        print("="*60 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\nGoodbye! 👋\n")
                    break
                
                if user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("\n[Conversation history cleared]\n")
                    continue
                
                if user_input.lower() == 'history':
                    if self.conversation_history:
                        print("\n--- Conversation History ---")
                        for i, (role, text) in enumerate(self.conversation_history, 1):
                            print(f"{i}. {role}: {text}")
                        print("----------------------------\n")
                    else:
                        print("\n[No conversation history]\n")
                    continue
                
                # Add user message to history
                self.conversation_history.append(("You", user_input))
                
                # Build prompt with conversation history (last few exchanges)
                context_length = min(3, len(self.conversation_history))
                context = self.conversation_history[-context_length:]
                
                prompt = ""
                for role, text in context:
                    prompt += f"{role}: {text}\n"
                prompt += "Bot:"
                
                # Generate response
                print("Bot: ", end="", flush=True)
                response = self.generate(
                    prompt,
                    max_new_tokens=150,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.95
                )
                
                # Clean up response
                response = response.split('\n')[0].strip()  # Take first line
                if not response:
                    response = "I'm not sure how to respond to that."
                
                print(response)
                print()  # Blank line
                
                # Add bot response to history
                self.conversation_history.append(("Bot", response))
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋\n")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.\n")


def main():
    """Main function to run the chatbot."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cogni-Mamba CLI Chatbot')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model checkpoint not found at {args.model}")
        print("Please train the model first using train.py")
        sys.exit(1)
    
    # Initialize and run chatbot
    chatbot = Chatbot(model_path=args.model, device=args.device)
    chatbot.chat()


if __name__ == '__main__':
    main()
