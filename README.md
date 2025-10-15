# Cogni-Mamba: Efficient CLI Chatbot with RAG 🤖

A lightweight, resource-efficient Large Language Model (LLM) with a command-line chatbot interface and **RAG (Retrieval-Augmented Generation)** capabilities. Built with PyTorch and optimized for minimal hardware requirements without sacrificing accuracy.

## ✨ Key Features

- **Efficient Architecture**: 168M parameters with Grouped-Query Attention (GQA)
- **Memory Optimized**: Mixed precision training (FP16), gradient accumulation
- **Low Resource Usage**: Runs on 4GB+ GPU or CPU
- **RAG Support**: Answer questions based on your documents (PDF, TXT, DOCX)
- **Multiple Datasets**: OpenAssistant, Dolly, Alpaca, TinyStories, Code datasets
- **Advanced Components**:
  - Rotary Positional Embeddings (RoPE)
  - SwiGLU activation
  - RMSNorm layer normalization
  - Flash Attention support
- **Interactive CLI**: Simple command-line chatbot interface with document Q&A

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Optional (for better RAG):
```bash
pip install sentence-transformers PyPDF2 python-docx
```

### 2. Train the Model

```bash
python train.py
```

Now trains on **OpenAssistant** (high-quality conversations).  
Training takes ~2-3 hours on GTX 1650 (4GB).

### 3. Chat with Your Bot

**Regular Chat:**
```bash
python chatbot.py
```

**RAG Chat (Document Q&A):**
```bash
python chatbot_rag.py
```

**With Document Pre-loaded:**
```bash
python chatbot_rag.py --document sample_document.txt
```

## 📁 Project Structure

```
Custom_LLM/
├── LLM_architecture_168.py   # Core model architecture
├── train.py                   # Training script (OpenAssistant)
├── chatbot.py                 # Simple CLI chatbot
├── chatbot_rag.py            # RAG-enabled chatbot ⭐ NEW
├── rag_pipeline.py           # RAG implementation ⭐ NEW
├── tokenizer_utils.py         # Tokenization utilities
├── data_loader.py             # Multi-dataset loader (updated)
├── config.json                # Model configuration (GTX 1650 optimized)
├── requirements.txt           # Python dependencies
├── sample_document.txt        # Example document for RAG
├── test_rag.py               # RAG test script
├── quick_start.md            # Detailed setup guide
├── RAG_GUIDE.md              # RAG usage guide ⭐ NEW
├── TRAINING_WITH_RAG.md      # Complete implementation guide ⭐ NEW
└── checkpoints/              # Saved model checkpoints
```

## 💻 Hardware Requirements

### Minimum (Current Configuration)
- GPU: 4GB VRAM (GTX 1650, GTX 1050 Ti) ⭐ Optimized
- RAM: 8GB
- Storage: 5GB

### Recommended
- GPU: 6GB+ VRAM (GTX 1660, RTX 3050)
- RAM: 16GB
- Storage: 10GB

### CPU-Only Mode
Supported but slower (training may take 6-8 hours).

## 🎯 Model Configuration

The model is configured in `config.json`:

```json
{
  "vocab_size": 50304,
  "dim": 1024,              // Model dimension
  "n_layers": 12,           // Transformer layers
  "n_heads": 16,            // Attention heads
  "n_kv_heads": 4,          // KV heads for GQA
  "hidden_dim": 2816,       // FFN hidden size
  "max_seq_len": 512,       // Max sequence length
  "batch_size": 4,
  "gradient_accumulation_steps": 8,
  "mixed_precision": true
}
```

## 📊 Training Features

### Memory Optimizations
- **Mixed Precision (FP16)**: Reduces memory by 50%
- **Gradient Accumulation**: Simulates larger batches
- **Weight Tying**: Shares embedding weights
- **Efficient Attention**: Grouped-Query Attention

### Training Techniques
- AdamW optimizer with weight decay
- Cosine learning rate schedule with warmup
- Gradient clipping for stability
- Automatic checkpointing

## 🎮 Using the Chatbot

### Regular Chatbot (chatbot.py)

```bash
You: Hello!
Bot: Hi! How can I help you today?

Commands:
  - 'quit' or 'exit': Exit the chatbot
  - 'clear': Clear conversation history
  - 'history': Show conversation history
```

### RAG Chatbot (chatbot_rag.py) ⭐ NEW

```bash
You: add sample_document.txt
✓ Document added: sample_document.txt

You: What is machine learning?
Bot: Based on the document, machine learning is a subset of AI...

Commands:
  - 'add <filepath>': Add document to knowledge base
  - 'docs': List loaded documents
  - 'save kb': Save knowledge base
  - 'load kb': Load knowledge base
  - 'quit', 'clear', 'history': Same as regular chatbot
```

### Generation Parameters

Adjust in `chatbot.py`:
- `temperature`: 0.7-1.0 (higher = more creative)
- `top_k`: 40-50 (smaller = more focused)
- `top_p`: 0.9-0.95 (nucleus sampling)
- `max_new_tokens`: 50-200 (response length)

## 📚 Datasets

### Default: OpenAssistant ⭐ NEW
- High-quality human conversations
- 161K samples
- Instruction-following format
- Best for chatbots

### Alternatives (in train.py):
- **Dolly-15k**: Instruction following (fast training)
- **Alpaca**: Q&A format (good quality)  
- **TinyStories**: Simple text (basic testing)
- **Code Search Net**: Python code (code tasks)

```python
# In train.py, change dataset_name
dataset_name='oasst'          # OpenAssistant (default)
dataset_name='dolly'          # Dolly-15k
dataset_name='alpaca'         # Alpaca
dataset_name='code_search_net' # Code
```

### Custom Data
Add your own training data in `data_loader.py`.

## 🔧 Troubleshooting

### Out of Memory
1. Reduce batch size to 2 in `config.json`
2. Reduce `max_seq_len` to 256
3. Reduce model size (fewer layers/smaller dim)

### Poor Responses
1. Train longer (more epochs)
2. Use more training data
3. Adjust generation temperature

### Installation Issues
```bash
# Install PyTorch first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Then install other dependencies
pip install transformers datasets tqdm accelerate
```

## 📈 Performance

### Training Speed
- GPU (RTX 3060): ~40 min for 3 epochs (10K samples)
- GPU (GTX 1060): ~60 min for 3 epochs
- CPU: 4-6 hours

### Model Size
- Parameters: 168M (0.17B)
- Disk size: ~650MB (FP32), ~325MB (FP16)
- Memory usage: ~1-2GB during inference

## 🛠️ Advanced Usage

### Fine-tuning
```bash
python train.py --checkpoint checkpoints/best_model.pt
```

### Inference Only
```python
from chatbot import Chatbot

bot = Chatbot(model_path='checkpoints/best_model.pt')
response = bot.generate("Your prompt here")
```

### Export Model
```python
# Save only weights
torch.save(model.state_dict(), 'model_weights.pt')
```

## 📖 Documentation

- **TRAINING_WITH_RAG.md** - Complete implementation guide (START HERE) ⭐
- **RAG_GUIDE.md** - RAG usage and best practices
- **quick_start.md** - Detailed setup instructions
- **IMPLEMENTATION_GUIDE.md** - Architecture details
- **QUICK_REFERENCE.txt** - Command reference card

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📄 License

See LICENSE file for details.

## 🙏 Acknowledgments

- Architecture inspired by modern LLMs (Llama, Mistral)
- Optimizations from Flash Attention and efficient transformers research
- Training techniques from various open-source projects

---

**Built with ❤️ for efficient AI**
custom LLM
## in development... stay tuned homies!
