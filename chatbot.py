"""
Simple Chatbot using SmolLM2-135M
An interactive chatbot with conversation history
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class SimpleChatbot:
    def __init__(self, checkpoint="HuggingFaceTB/SmolLM2-1.7B", device=None):
        """
        Initialize the chatbot with a pre-trained model
        
        Args:
            checkpoint: Model checkpoint from HuggingFace
            device: 'cuda' for GPU or 'cpu' for CPU (auto-detected if None)
        """
        print("🤖 Initializing chatbot...")
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"   Using device: {self.device}")
        
        # Load tokenizer and model
        print(f"   Loading model: {checkpoint}")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(self.device)
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.conversation_history = []
        print("✓ Chatbot ready!\n")
    
    def generate_response(self, user_input, max_length=100, temperature=0.7, top_p=0.9):
        """
        Generate a response to user input
        
        Args:
            user_input: User's message
            max_length: Maximum length of generated response
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated response text
        """
        # Build prompt with conversation history
        if self.conversation_history:
            # Include last few exchanges for context
            context = "\n".join(self.conversation_history[-6:])  # Last 3 exchanges
            prompt = f"{context}\nUser: {user_input}\nAssistant:"
        else:
            prompt = f"User: {user_input}\nAssistant:"
        
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=min(len(inputs[0]) + max_length, 512),
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Decode and extract response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "Assistant:" in full_response:
            response = full_response.split("Assistant:")[-1].strip()
            # Stop at next "User:" if present
            if "User:" in response:
                response = response.split("User:")[0].strip()
        else:
            response = full_response[len(prompt):].strip()
        
        return response
    
    def chat(self, user_input):
        """
        Process user input and return response
        
        Args:
            user_input: User's message
            
        Returns:
            Chatbot's response
        """
        # Generate response
        response = self.generate_response(user_input)
        
        # Update conversation history
        self.conversation_history.append(f"User: {user_input}")
        self.conversation_history.append(f"Assistant: {response}")
        
        return response
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("🗑️  Conversation history cleared.")
    
    def run_interactive(self):
        """
        Run interactive chatbot session
        """
        print("=" * 60)
        print("       🤖 SIMPLE CHATBOT - SmolLM2-135M")
        print("=" * 60)
        print("\nCommands:")
        print("  - Type your message to chat")
        print("  - Type 'clear' to clear conversation history")
        print("  - Type 'quit' or 'exit' to end the conversation")
        print("\n" + "=" * 60 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                # Check for commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye! Thanks for chatting!")
                    break
                
                if user_input.lower() == 'clear':
                    self.clear_history()
                    continue
                
                if not user_input:
                    continue
                
                # Generate and display response
                print("🤖 Bot: ", end="", flush=True)
                response = self.chat(user_input)
                print(response + "\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thanks for chatting!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


def main():
    """Main function to run the chatbot"""
    # Create chatbot instance
    # Change device to "cpu" if you don't have a GPU
    chatbot = SimpleChatbot(device="cuda")  # Change to "cuda" if you have GPU
    
    # Run interactive session
    chatbot.run_interactive()


if __name__ == "__main__":
    main()
