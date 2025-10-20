"""
CLI-based Chatbot Interface for LLM-200M
Interactive command-line interface for chatting with the trained model
"""

import torch
from transformers import AutoTokenizer
import os
import sys
from typing import Optional
# import readline  # For better input handling
from colorama import init, Fore, Style

from model import LLM200M

# Initialize colorama for colored terminal output
init(autoreset=True)


class ChatBot:
    """Interactive CLI chatbot"""
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1
    ):
        """
        Initialize the chatbot
        
        Args:
            model_path: path to trained model checkpoint
            device: 'cuda' or 'cpu'
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling parameter
            top_p: nucleus sampling parameter
            repetition_penalty: penalty for token repetition
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        print(f"{Fore.CYAN}Loading tokenizer...{Style.RESET_ALL}")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"{Fore.CYAN}Loading model from {model_path}...{Style.RESET_ALL}")
        self.model = LLM200M(vocab_size=50257)  # GPT-2 vocab size
        
        # Load checkpoint
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"{Fore.GREEN}✓ Model loaded successfully!{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}✗ Model checkpoint not found at {model_path}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Using untrained model (for testing only){Style.RESET_ALL}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Generation parameters
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        
        # Conversation history
        self.conversation_history = []
        
        print(f"{Fore.GREEN}✓ Chatbot ready on {self.device}!{Style.RESET_ALL}\n")
    
    def format_prompt(self, user_input: str, use_history: bool = True) -> str:
        """
        Format the prompt with instruction template
        
        Args:
            user_input: user's message
            use_history: whether to include conversation history
            
        Returns:
            formatted prompt
        """
        if use_history and self.conversation_history:
            # Include previous conversation
            context = "\n".join([
                f"User: {h['user']}\nAssistant: {h['assistant']}"
                for h in self.conversation_history[-3:]  # Last 3 exchanges
            ])
            prompt = f"{context}\nUser: {user_input}\nAssistant:"
        else:
            prompt = f"### Instruction:\n{user_input}\n\n### Response:"
        
        return prompt
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate response for the given prompt
        
        Args:
            prompt: formatted prompt
            
        Returns:
            generated response
        """
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Check if input is too long
        if input_ids.shape[1] > self.model.max_seq_len - self.max_new_tokens:
            input_ids = input_ids[:, -(self.model.max_seq_len - self.max_new_tokens):]
            print(f"{Fore.YELLOW}⚠ Input truncated to fit context window{Style.RESET_ALL}")
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty
            )
        
        # Decode
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract only the new response
        response = generated_text[len(prompt):].strip()
        
        # Clean up response (stop at newlines or special markers)
        if '\n\n' in response:
            response = response.split('\n\n')[0]
        if '###' in response:
            response = response.split('###')[0]
        
        return response.strip()
    
    def chat(self):
        """Main chat loop"""
        self.print_welcome()
        
        while True:
            try:
                # Get user input
                user_input = input(f"{Fore.BLUE}You: {Style.RESET_ALL}").strip()
                
                # Handle commands
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print(f"{Fore.YELLOW}Goodbye! 👋{Style.RESET_ALL}")
                    break
                
                if user_input.lower() == 'clear':
                    self.conversation_history = []
                    os.system('cls' if os.name == 'nt' else 'clear')
                    self.print_welcome()
                    continue
                
                if user_input.lower() == 'help':
                    self.print_help()
                    continue
                
                if user_input.lower().startswith('temp '):
                    try:
                        self.temperature = float(user_input.split()[1])
                        print(f"{Fore.GREEN}✓ Temperature set to {self.temperature}{Style.RESET_ALL}")
                    except:
                        print(f"{Fore.RED}✗ Invalid temperature value{Style.RESET_ALL}")
                    continue
                
                if user_input.lower().startswith('length '):
                    try:
                        self.max_new_tokens = int(user_input.split()[1])
                        print(f"{Fore.GREEN}✓ Max length set to {self.max_new_tokens}{Style.RESET_ALL}")
                    except:
                        print(f"{Fore.RED}✗ Invalid length value{Style.RESET_ALL}")
                    continue
                
                if not user_input:
                    continue
                
                # Format prompt and generate response
                prompt = self.format_prompt(user_input)
                
                print(f"{Fore.GREEN}Assistant: {Style.RESET_ALL}", end='', flush=True)
                response = self.generate_response(prompt)
                print(response)
                
                # Add to history
                self.conversation_history.append({
                    'user': user_input,
                    'assistant': response
                })
                
                print()  # Empty line for readability
                
            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Interrupted. Type 'exit' to quit.{Style.RESET_ALL}")
                continue
            except Exception as e:
                print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
                continue
    
    def print_welcome(self):
        """Print welcome message"""
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'LLM-200M Chatbot'.center(70)}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Type 'help' for commands, 'exit' to quit{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")
    
    def print_help(self):
        """Print help message"""
        print(f"\n{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'Available Commands'.center(70)}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}help{Style.RESET_ALL}          - Show this help message")
        print(f"{Fore.GREEN}clear{Style.RESET_ALL}         - Clear conversation history")
        print(f"{Fore.GREEN}temp <val>{Style.RESET_ALL}    - Set temperature (0.1-2.0, current: {self.temperature})")
        print(f"{Fore.GREEN}length <val>{Style.RESET_ALL}  - Set max response length (current: {self.max_new_tokens})")
        print(f"{Fore.GREEN}exit/quit{Style.RESET_ALL}     - Exit the chatbot")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM-200M CLI Chatbot')
    parser.add_argument(
        '--model-path',
        type=str,
        default='training_outputs/llm_200m_final.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        choices=['pretrain', 'finetune', 'final'],
        default='final',
        help='Which checkpoint to use'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to run on (cuda/cpu)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature (default: 0.8)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=256,
        help='Maximum response length (default: 256)'
    )
    
    args = parser.parse_args()
    
    # Determine model path
    if args.checkpoint == 'pretrain':
        model_path = 'checkpoints/pretrain_best.pt'
    elif args.checkpoint == 'finetune':
        model_path = 'checkpoints/finetune_best.pt'
    else:
        model_path = args.model_path
    
    # Initialize chatbot
    chatbot = ChatBot(
        model_path=model_path,
        device=args.device,
        max_new_tokens=args.max_length,
        temperature=args.temperature
    )
    
    # Start chatting
    chatbot.chat()


if __name__ == "__main__":
    main()
