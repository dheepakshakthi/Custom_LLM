# LLM-200M: CLI-Based Chatbot with Custom Language Model

A 200M parameter language model built from scratch, inspired by SmolLM-135M, designed for better reasoning and performance. The model is pre-trained on BookCorpus and fine-tuned on OpenCoder-LLM instruction data.

## 🌟 Features

- **200M Parameter Architecture**: Transformer-based model with:
  - 768 embedding dimension
  - 24 transformer layers for deep reasoning
  - 12 attention heads
  - RoPE positional embeddings
  - SwiGLU activation functions
  - Flash Attention support (when available)

- **Unified Training Pipeline**: Single script for both pre-training and fine-tuning
  - Pre-training on BookCorpus dataset
  - Fine-tuning on OpenCoder-LLM/opc-sft-stage2
  - Automatic hyperparameter optimization
  - Prevents overfitting and underfitting

- **CLI Chatbot Interface**: Interactive command-line chat interface with:
  - Colored output for better readability
  - Conversation history
  - Adjustable generation parameters
  - Multiple commands for customization

- **Comprehensive Monitoring**: 
  - Training metrics tracking (loss, perplexity, learning rate)
  - Automatic checkpointing
  - Early stopping to prevent overfitting
  - Statistics saved in JSON format

## 📁 Project Structure

```
Custom_LLM/
├── model.py              # 200M parameter LLM architecture
├── data_processing.py    # Data loading and preprocessing
├── train.py             # Unified training script
├── chatbot.py           # CLI chatbot interface
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── archive/            # Dataset folder
│   └── BookCorpus3.csv # Pre-training dataset
├── checkpoints/        # Model checkpoints (created during training)
├── training_stats/     # Training statistics (created during training)
└── training_outputs/   # Final trained models (created during training)
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended, but CPU works too)
- 16GB+ RAM
- 50GB+ free disk space

### Setup

1. **Clone or navigate to the project directory**

2. **Create a virtual environment (recommended)**
```powershell
python -m venv master
.\master\Scripts\activate
```

3. **Install dependencies**
```powershell
pip install -r requirements.txt
```

## 📚 Dataset Preparation

### Pre-training Dataset (BookCorpus)
The BookCorpus dataset should be in the `archive/` folder as `BookCorpus3.csv`. 

**Note**: If your CSV file has a different structure, the code will automatically:
- Try to find a 'text' column
- Fall back to the first column if no 'text' column exists
- Create a small dummy dataset if the file can't be loaded (for testing)

### Fine-tuning Dataset (OpenCoder)
The OpenCoder dataset will be automatically downloaded from HuggingFace when you run the training script.

## 🎓 Training

### Start Training

Run the complete training pipeline (pre-training + fine-tuning):

```powershell
python train.py
```

### Training Configuration

The training script uses optimized hyperparameters to prevent overfitting/underfitting:

**Pre-training:**
- Epochs: 3
- Learning rate: 3e-4 with warmup
- Weight decay: 0.01 (L2 regularization)
- Gradient clipping: 1.0
- Dropout: 0.1

**Fine-tuning:**
- Epochs: 2
- Learning rate: 1e-4 with warmup
- Early stopping: patience of 5 evaluations
- Same regularization as pre-training

### Training Outputs

During training, the following will be created:

1. **Checkpoints** (`checkpoints/`):
   - `pretrain_best.pt` - Best pre-training checkpoint
   - `finetune_best.pt` - Best fine-tuning checkpoint
   - `*_latest.pt` - Latest checkpoints

2. **Statistics** (`training_stats/`):
   - `training_metrics.json` - Complete training metrics

3. **Final Model** (`training_outputs/`):
   - `llm_200m_final.pt` - Final trained model

### Monitoring Training

The training script provides real-time monitoring:
- Loss and perplexity metrics
- Learning rate schedule
- Progress bars with ETA
- Validation metrics
- Early stopping notifications

## 💬 Using the Chatbot

### Basic Usage

After training, start the chatbot:

```powershell
python chatbot.py
```

### Advanced Usage

```powershell
# Use specific checkpoint
python chatbot.py --checkpoint finetune

# Use pre-trained checkpoint
python chatbot.py --checkpoint pretrain

# Custom model path
python chatbot.py --model-path path/to/model.pt

# Adjust temperature (0.1-2.0, higher = more creative)
python chatbot.py --temperature 0.9

# Adjust response length
python chatbot.py --max-length 512

# Force CPU usage
python chatbot.py --device cpu
```

### Chatbot Commands

While in the chatbot:
- `help` - Show available commands
- `clear` - Clear conversation history
- `temp <value>` - Set temperature (e.g., `temp 0.7`)
- `length <value>` - Set max response length (e.g., `length 200`)
- `exit` or `quit` - Exit the chatbot

### Example Conversation

```
You: What is Python?
Assistant: Python is a high-level programming language known for its simplicity and readability...

You: Write a function to reverse a string
Assistant: Here's a function to reverse a string:

def reverse_string(s):
    return s[::-1]

You: exit
Goodbye! 👋
```

## 🔧 Customization

### Modify Model Architecture

Edit `model.py` to change model parameters:

```python
model = LLM200M(
    vocab_size=32000,      # Vocabulary size
    embed_dim=768,         # Embedding dimension
    num_layers=24,         # Number of transformer layers
    num_heads=12,          # Number of attention heads
    ff_dim=3072,           # Feed-forward dimension
    max_seq_len=2048,      # Maximum sequence length
    dropout=0.1            # Dropout rate
)
```

### Modify Training Hyperparameters

Edit `train.py`:

```python
trainer = Trainer(
    pretrain_epochs=3,              # Pre-training epochs
    finetune_epochs=2,              # Fine-tuning epochs
    pretrain_lr=3e-4,               # Pre-training learning rate
    finetune_lr=1e-4,               # Fine-tuning learning rate
    batch_size=8,                   # Batch size
    gradient_accumulation_steps=4,  # Gradient accumulation
    weight_decay=0.01,              # L2 regularization
    early_stopping_patience=5       # Early stopping patience
)
```

## 📊 Model Architecture Details

### Key Features:

1. **Rotary Position Embeddings (RoPE)**
   - Better position encoding than absolute positional embeddings
   - Enables length extrapolation

2. **SwiGLU Activation**
   - Superior to GELU and ReLU
   - Improves model performance

3. **Pre-Layer Normalization**
   - More stable training for deep networks
   - Better gradient flow

4. **Flash Attention**
   - Memory-efficient attention computation
   - Faster training and inference

5. **Weight Tying**
   - Shares token embeddings with output layer
   - Reduces parameters while maintaining performance

### Parameter Count

```
Token Embeddings:    24,576,000
24 Transformer Layers: ~170,000,000
Output Layer:        (tied with embeddings)
Total:              ~200,000,000 parameters
```

## 🛡️ Preventing Overfitting/Underfitting

The training pipeline includes multiple techniques:

**Regularization:**
- L2 weight decay (0.01)
- Dropout (0.1)
- Gradient clipping (1.0)

**Training Strategy:**
- Learning rate warmup
- Cosine annealing schedule
- Early stopping
- Validation monitoring

**Data Strategy:**
- Large pre-training corpus
- Diverse fine-tuning data
- Proper train/val split

## 📈 Training Statistics

All training statistics are saved in `training_stats/training_metrics.json`:

```json
{
  "pretrain": {
    "losses": [...],
    "perplexities": [...],
    "learning_rates": [...],
    "timestamps": [...]
  },
  "finetune": {
    "train_losses": [...],
    "val_losses": [...],
    "train_perplexities": [...],
    "val_perplexities": [...],
    "learning_rates": [...],
    "timestamps": [...]
  }
}
```

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in `train.py`
- Increase `gradient_accumulation_steps`
- Reduce `max_seq_len` in model config

### Dataset Loading Issues
- Ensure `BookCorpus3.csv` is in the `archive/` folder
- Check CSV format (should have a 'text' column)
- The code will create dummy data if loading fails (for testing)

### Slow Training
- Use GPU instead of CPU
- Reduce dataset size for testing
- Decrease model size (reduce `num_layers` or `embed_dim`)

### Model Not Generating Good Responses
- Ensure training completed successfully
- Try different checkpoints (`pretrain_best.pt` vs `finetune_best.pt`)
- Adjust generation parameters (temperature, top_k, top_p)

## 📝 Notes

- **Training Time**: Full training can take several hours to days depending on hardware
- **GPU Recommended**: Training on CPU will be extremely slow
- **Dataset**: If the BookCorpus dataset isn't correct, the code will use dummy data for testing
- **Checkpointing**: Models are saved regularly to prevent data loss

## 🤝 Contributing

Feel free to:
- Report issues
- Suggest improvements
- Add new features
- Optimize the code

## 📄 License

This project is for educational purposes. The model architecture is inspired by SmolLM-135M.

## 🙏 Acknowledgments

- **SmolLM-135M**: Architecture inspiration
- **OpenCoder-LLM**: Fine-tuning dataset
- **BookCorpus**: Pre-training dataset
- **HuggingFace**: Transformers library and datasets

## 📧 Contact

For questions or issues, please open an issue in the repository.

---

**Happy Training! 🚀**
