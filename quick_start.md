# Cogni-Mamba Chatbot - Quick Start Guide

This guide will help you set up and run your CLI chatbot with minimal hardware requirements.

## 🚀 Quick Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

If you encounter issues, install packages individually:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets tqdm accelerate
```

### 2. Train the Model

The training script is configured for minimal resource usage:
- **Mixed Precision Training**: Uses FP16 to reduce memory by ~50%
- **Gradient Accumulation**: Simulates larger batch sizes
- **Small Model**: ~168M parameters (0.17B)
- **Limited Dataset**: Uses 10K samples for quick training

Start training:
```bash
python train.py
```

**Hardware Requirements:**
- Minimum: 6GB GPU RAM (GTX 1060, RTX 3050)
- Recommended: 8GB+ GPU RAM
- CPU fallback available (slower)

**Training Time:**
- GPU: ~30-60 minutes for 3 epochs
- CPU: Several hours

### 3. Run the Chatbot

After training completes:
```bash
python chatbot.py
```

Or specify a custom checkpoint:
```bash
python chatbot.py --model checkpoints/best_model.pt --device cuda
```

## 🎯 Features

### Memory Optimizations
1. **Mixed Precision (FP16)**: Reduces memory usage by 50%
2. **Gradient Accumulation**: Effective batch size without memory overhead
3. **Grouped-Query Attention**: More efficient than standard attention
4. **Weight Tying**: Shares embedding and output layer weights

### Training Optimizations
- AdamW optimizer with weight decay
- Cosine learning rate schedule with warmup
- Gradient clipping for stability
- Automatic checkpointing

### Chatbot Features
- Interactive CLI interface
- Conversation history tracking
- Top-k and nucleus (top-p) sampling
- Temperature control for creativity

## 📊 Model Configuration

Edit `config.json` to adjust model size:

```json
{
  "vocab_size": 50304,
  "dim": 1024,              // Embedding dimension
  "n_layers": 12,           // Number of transformer layers
  "n_heads": 16,            // Attention heads
  "n_kv_heads": 4,          // KV heads (for GQA)
  "hidden_dim": 2816,       // FFN dimension
  "max_seq_len": 512,       // Maximum sequence length
  "batch_size": 4,          // Batch size per GPU
  "gradient_accumulation_steps": 8  // Effective batch = 32
}
```

**For even lower memory:**
- Reduce `dim` to 768
- Reduce `n_layers` to 8-10
- Reduce `hidden_dim` to 2048
- Reduce `max_seq_len` to 256

## 🔧 Troubleshooting

### Out of Memory Errors

1. **Reduce batch size** in `config.json`:
   ```json
   "batch_size": 2
   ```

2. **Reduce sequence length**:
   ```json
   "max_seq_len": 256
   ```

3. **Use CPU** (slower but works):
   ```bash
   python train.py --device cpu
   ```

### Model Not Training Well

1. **Increase training data**: Modify `max_train_samples` in `train.py`
2. **Train longer**: Increase `max_epochs` in `config.json`
3. **Adjust learning rate**: Try 1e-4 or 5e-4

### Chatbot Responses Are Poor

1. **Train longer**: More epochs improve quality
2. **Use better dataset**: Switch to code_search_net or custom data
3. **Adjust generation parameters**:
   ```python
   response = self.generate(
       prompt,
       temperature=0.7,  # Lower = more focused
       top_k=40,         # Smaller = less random
       top_p=0.9         # Lower = more focused
   )
   ```

## 📈 Dataset Options

### TinyStories (Default)
- Simple, clean text
- Good for quick training
- Limited complexity

### Code Search Net (for Phi-1 style)
- Python code snippets
- Better for coding tasks
- Larger vocabulary

To switch datasets, edit `train.py`:
```python
train_loader, val_loader = create_dataloaders(
    tokenizer=tokenizer,
    config=config,
    dataset_name='code_search_net',  # Change here
    max_train_samples=10000,
    max_val_samples=1000
)
```

### Custom Dataset

Add your own data in `data_loader.py`:
```python
def _get_sample_data(self):
    return [
        "Your custom training text here",
        "More examples...",
        # Add hundreds/thousands of examples
    ]
```

## 🎮 Chatbot Commands

While chatting:
- `quit` or `exit` - Exit the chatbot
- `clear` - Clear conversation history
- `history` - View conversation history

## 📝 Example Conversation

```
You: Hello! How are you?
Bot: I'm doing well, thank you for asking!

You: What can you help me with?
Bot: I can chat with you about various topics and answer questions.

You: Tell me about neural networks
Bot: Neural networks are computational models inspired by the human brain...
```

## 🔬 Advanced Usage

### Fine-tuning on Custom Data

1. Prepare your text data
2. Modify `data_loader.py` to load your data
3. Run training with adjusted hyperparameters

### Exporting the Model

Save just the model weights:
```python
torch.save(model.state_dict(), 'model_weights.pt')
```

### Inference Optimization

For faster inference:
- Use `torch.compile()` (PyTorch 2.0+)
- Reduce `max_new_tokens` in generation
- Use greedy decoding (temperature=0)

## 📚 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transformers Library](https://huggingface.co/docs/transformers/)
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)

## 💡 Tips for Best Results

1. **Start small**: Train on a small dataset first to verify everything works
2. **Monitor training**: Watch the loss curves to ensure learning
3. **Experiment**: Try different temperatures and sampling parameters
4. **Save often**: The trainer saves checkpoints automatically
5. **Use GPU**: Training on GPU is 10-50x faster than CPU

Enjoy your chatbot! 🤖✨
