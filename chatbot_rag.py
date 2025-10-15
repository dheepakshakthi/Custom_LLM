"""
Enhanced CLI Chatbot with RAG (Retrieval-Augmented Generation)
Allows uploading documents and answering questions based on them.
"""
import torch
from pathlib import Path
import sys
import os

from LLM_architecture_168 import CogniMamba, ModelArgs
from tokenizer_utils import GPT2Tokenizer, SimpleTokenizer
from rag_pipeline import RAGChatbot

class EnhancedChatbot:
    """CLI Chatbot with RAG capabilities."""
    
    def __init__(self, model_path='checkpoints/best_model.pt', device='cuda'):
        """Initialize the chatbot with RAG support."""
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
        
        print(f"Model loaded! ({self.model.get_num_params() / 1e6:.2f}M parameters)")
        
        # Initialize tokenizer
        try:
            self.tokenizer = GPT2Tokenizer()
            print("Using GPT2Tokenizer")
        except ImportError:
            self.tokenizer = SimpleTokenizer()
            print("Using SimpleTokenizer")
        
        # Initialize RAG chatbot
        self.rag_bot = RAGChatbot(self.model, self.tokenizer, self.device)
        
        self.conversation_history = []
        self.documents_loaded = []
    
    def add_document(self, file_path: str):
        """Add a document to the knowledge base."""
        try:
            self.rag_bot.add_document(file_path)
            self.documents_loaded.append(Path(file_path).name)
            print(f"\n✓ Document added: {Path(file_path).name}\n")
            return True
        except Exception as e:
            print(f"\n✗ Error adding document: {e}\n")
            return False
    
    def chat(self):
        """Start an interactive chat session with RAG support."""
        print("\n" + "="*70)
        print("🤖 Cogni-Mamba Chatbot with RAG (Document Q&A)")
        print("="*70)
        print("Type your message and press Enter to chat.")
        print("\nCommands:")
        print("  - 'quit' or 'exit'     : Exit the chatbot")
        print("  - 'clear'              : Clear conversation history")
        print("  - 'history'            : Show conversation history")
        print("  - 'add <filepath>'     : Add a document to knowledge base")
        print("  - 'docs'               : List loaded documents")
        print("  - 'save kb'            : Save knowledge base")
        print("  - 'load kb'            : Load knowledge base")
        print("="*70)
        
        if self.documents_loaded:
            print(f"\n📚 Documents loaded: {', '.join(self.documents_loaded)}")
        else:
            print("\n💡 Tip: Use 'add <filepath>' to upload a document for Q&A")
        
        print()
        
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
                
                if user_input.lower().startswith('add '):
                    file_path = user_input[4:].strip()
                    self.add_document(file_path)
                    continue
                
                if user_input.lower() == 'docs':
                    if self.documents_loaded:
                        print("\n📚 Loaded documents:")
                        for i, doc in enumerate(self.documents_loaded, 1):
                            print(f"  {i}. {doc}")
                        print()
                    else:
                        print("\n[No documents loaded]\n")
                    continue
                
                if user_input.lower() == 'save kb':
                    self.rag_bot.save_knowledge_base('knowledge_base.json')
                    print("\n✓ Knowledge base saved\n")
                    continue
                
                if user_input.lower() == 'load kb':
                    try:
                        self.rag_bot.load_knowledge_base('knowledge_base.json')
                        print("\n✓ Knowledge base loaded\n")
                    except FileNotFoundError:
                        print("\n✗ No saved knowledge base found\n")
                    continue
                
                # Add user message to history
                self.conversation_history.append(("You", user_input))
                
                # Generate response with RAG
                print("Bot: ", end="", flush=True)
                
                if self.documents_loaded:
                    # Use RAG if documents are loaded
                    response = self.rag_bot.generate_with_context(
                        user_input,
                        max_new_tokens=150,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.9
                    )
                else:
                    # Use regular generation
                    # Build prompt with conversation history
                    context_length = min(3, len(self.conversation_history))
                    context = self.conversation_history[-context_length:]
                    
                    prompt = ""
                    for role, text in context:
                        prompt += f"{role}: {text}\n"
                    prompt += "Bot:"
                    
                    # Generate
                    input_ids = self.tokenizer.encode(prompt)
                    input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
                    
                    generated = input_ids
                    
                    with torch.no_grad():
                        for _ in range(150):
                            if generated.shape[1] > self.model.params.max_seq_len:
                                input_chunk = generated[:, -self.model.params.max_seq_len:]
                            else:
                                input_chunk = generated
                            
                            outputs = self.model(input_chunk)
                            logits = outputs[:, -1, :] / 0.7
                            
                            # Top-k
                            if 50 > 0:
                                indices_to_remove = logits < torch.topk(logits, 50)[0][..., -1, None]
                                logits[indices_to_remove] = float('-inf')
                            
                            probs = torch.softmax(logits, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                            generated = torch.cat([generated, next_token], dim=1)
                            
                            if hasattr(self.tokenizer, 'tokenizer') and next_token.item() == self.tokenizer.tokenizer.eos_token_id:
                                break
                    
                    generated_text = self.tokenizer.decode(generated[0].tolist())
                    prompt_decoded = self.tokenizer.decode(input_ids[0].tolist())
                    if generated_text.startswith(prompt_decoded):
                        response = generated_text[len(prompt_decoded):].strip()
                    else:
                        response = generated_text.strip()
                    
                    response = response.split('\n')[0].strip()
                    if not response:
                        response = "I'm not sure how to respond to that."
                
                print(response)
                print()
                
                # Add bot response to history
                self.conversation_history.append(("Bot", response))
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋\n")
                break
            except Exception as e:
                print(f"\nError: {e}")
                print("Please try again.\n")


def main():
    """Main function to run the RAG chatbot."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cogni-Mamba CLI Chatbot with RAG')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--document', type=str, default=None,
                        help='Path to document to load on startup')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"Error: Model checkpoint not found at {args.model}")
        print("Please train the model first using train.py")
        sys.exit(1)
    
    # Initialize chatbot
    chatbot = EnhancedChatbot(model_path=args.model, device=args.device)
    
    # Load document if provided
    if args.document:
        chatbot.add_document(args.document)
    
    # Start chat
    chatbot.chat()


if __name__ == '__main__':
    main()
