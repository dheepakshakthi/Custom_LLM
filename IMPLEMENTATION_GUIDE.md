# Cogni-Mamba CLI Chatbot - Complete Implementation Guide

## 📋 Overview

You now have a complete CLI chatbot implementation with:
- **168M parameter LLM** with state-of-the-art architecture
- **Memory-efficient training** (runs on 6GB+ GPU)
- **Interactive CLI interface**
- **Flexible dataset support** (Phi-1 style or custom)

## 🗂️ Files Created

### Core Files
1. **LLM_architecture_168.py** - Your existing model architecture
2. **config.json** - Model and training configuration
3. **train.py** - Optimized training script
4. **chatbot.py** - Interactive CLI chatbot
5. **tokenizer_utils.py** - Tokenization utilities
6. **data_loader.py** - Efficient data loading

### Helper Files
7. **test_model.py** - Quick architecture test
8. **demo.py** - Quick demo script
9. **setup.ps1** - Automated setup script
10. **requirements.txt** - Python dependencies
11. **quick_start.md** - Detailed guide
12. **README.md** - Updated project documentation

## 🚀 Getting Started (3 Steps)

### Step 1: Setup Environment

**Option A: Automated (Recommended)**
```powershell
.\setup.ps1
```

**Option B: Manual**
```powershell
# Activate virtual environment
.\master\Scripts\Activate.ps1

# Install PyTorch with CUDA (for GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or PyTorch CPU-only
pip install torch torchvision torchaudio

# Install other dependencies
pip install transformers datasets tqdm accelerate
```

### Step 2: Test the Model
```powershell
python test_model.py
```

This verifies:
- ✓ Configuration loads correctly
- ✓ Model architecture works
- ✓ Forward pass succeeds
- ✓ GPU compatibility (if available)
- ✓ Gradient flow is correct

### Step 3: Train the Model
```powershell
python train.py
```

**What happens during training:**
- Loads TinyStories dataset (or code_search_net for Phi-1 style)
- Trains for 3 epochs (~30-60 min on GPU)
- Saves checkpoints every 1000 steps
- Saves best model based on validation loss
- Uses mixed precision (FP16) for efficiency

## 💬 Using the Chatbot

### Start Interactive Chat
```powershell
python chatbot.py
```

### Quick Demo
```powershell
python demo.py
```

### Commands During Chat
- `quit` or `exit` - Exit chatbot
- `clear` - Clear conversation history
- `history` - View conversation history

## ⚙️ Configuration Guide

### Minimal Resources (4GB GPU)
Edit `config.json`:
```json
{
  "dim": 768,
  "n_layers": 8,
  "n_heads": 12,
  "n_kv_heads": 3,
  "hidden_dim": 2048,
  "max_seq_len": 256,
  "batch_size": 2,
  "gradient_accumulation_steps": 16
}
```

### Balanced (6-8GB GPU) - Current Default
```json
{
  "dim": 1024,
  "n_layers": 12,
  "n_heads": 16,
  "n_kv_heads": 4,
  "hidden_dim": 2816,
  "max_seq_len": 512,
  "batch_size": 4,
  "gradient_accumulation_steps": 8
}
```

### High Performance (12GB+ GPU)
```json
{
  "dim": 1280,
  "n_layers": 16,
  "n_heads": 20,
  "n_kv_heads": 5,
  "hidden_dim": 3584,
  "max_seq_len": 1024,
  "batch_size": 8,
  "gradient_accumulation_steps": 4
}
```

## 📊 Memory Usage Guide

### Training Memory (FP16)
- **Minimal config**: ~3-4GB
- **Balanced config**: ~5-6GB
- **High performance**: ~10-12GB

### Inference Memory
- **Minimal config**: ~800MB
- **Balanced config**: ~1.5GB
- **High performance**: ~3GB

## 🎯 Training on Different Datasets

### TinyStories (Default - Simple Text)
```python
# In train.py
dataset_name='tinystories'
max_train_samples=10000
```

### Code Search Net (Phi-1 Style - Code)
```python
# In train.py
dataset_name='code_search_net'
max_train_samples=20000
```

### Custom Dataset
Edit `data_loader.py`:
```python
def _get_sample_data(self):
    return [
        "Your training data here",
        "More examples...",
        # Add 1000+ examples
    ]
```

Or load from file:
```python
with open('your_data.txt', 'r', encoding='utf-8') as f:
    data = f.readlines()
return [line.strip() for line in data if line.strip()]
```

## 🔧 Optimization Techniques Used

### Memory Optimizations
1. **Mixed Precision (FP16)**: 50% memory reduction
2. **Gradient Accumulation**: Simulate larger batches
3. **Grouped-Query Attention**: Fewer KV heads
4. **Weight Tying**: Share embedding/output weights
5. **Flash Attention**: Efficient attention computation

### Training Optimizations
1. **AdamW Optimizer**: Better regularization
2. **Cosine LR Schedule**: Smooth learning rate decay
3. **Warmup Steps**: Stable training start
4. **Gradient Clipping**: Prevent exploding gradients
5. **Automatic Checkpointing**: Save progress regularly

### Inference Optimizations
1. **Batch Size 1**: Minimal memory
2. **Top-k/Top-p Sampling**: Quality generation
3. **Temperature Control**: Creativity tuning
4. **Torch.no_grad()**: Disable gradients

## 📈 Expected Results

### Training Progress
- **Initial Loss**: 8-10
- **After 1 epoch**: 4-6
- **After 3 epochs**: 2-4
- **Best models**: 1.5-3

### Chat Quality
- **After 1 epoch**: Basic responses, some coherence
- **After 3 epochs**: Good responses, decent quality
- **With more data/epochs**: High quality, context-aware

### Generation Quality Factors
1. **Training data quality** (most important)
2. **Number of epochs** (more = better, up to a point)
3. **Model size** (larger = better)
4. **Generation parameters** (temperature, top_k, top_p)

## 🐛 Troubleshooting

### Issue: Out of Memory

**Solutions:**
1. Reduce batch_size in config.json
2. Reduce max_seq_len
3. Reduce model dimensions
4. Enable gradient_checkpointing (advanced)
5. Use CPU (slower but works)

### Issue: Model Not Training (Loss Not Decreasing)

**Solutions:**
1. Check learning rate (try 1e-4 or 5e-4)
2. Increase training data
3. Check data quality
4. Reduce model size (easier to train)
5. Increase warmup_steps

### Issue: Poor Chat Responses

**Solutions:**
1. Train longer (more epochs)
2. Use better/more training data
3. Adjust generation parameters:
   - Lower temperature (0.6-0.7) = more focused
   - Higher temperature (0.9-1.0) = more creative
   - Adjust top_k (20-50)
   - Adjust top_p (0.85-0.95)

### Issue: Installation Problems

**Solutions:**
1. Install PyTorch first separately
2. Use conda instead of pip
3. Check Python version (3.8+)
4. Update pip: `python -m pip install --upgrade pip`

### Issue: Slow Training

**Solutions:**
1. Enable CUDA (check with `torch.cuda.is_available()`)
2. Reduce dataset size (max_train_samples)
3. Use smaller max_seq_len
4. Ensure num_workers=0 on Windows

## 🎓 Understanding the Architecture

### Key Components

1. **Rotary Positional Embeddings (RoPE)**
   - Better than learned positions
   - Captures relative distances
   - Used by Llama, Mistral, etc.

2. **Grouped-Query Attention (GQA)**
   - Balance between MHA and MQA
   - Fewer KV heads than Q heads
   - 2-3x faster than standard attention

3. **SwiGLU Activation**
   - Gating mechanism
   - Better than ReLU/GELU
   - Used in modern LLMs

4. **RMSNorm**
   - Simpler than LayerNorm
   - Faster computation
   - Same quality

5. **Weight Tying**
   - Share input/output embeddings
   - Reduces parameters ~20%
   - No quality loss

### Training Flow

```
Data → Tokenize → Batch → Model Forward → Loss → Backward → Update Weights
                                           ↓
                                    Accumulate Gradients
                                    (every N steps)
```

### Inference Flow

```
User Input → Tokenize → Model Forward → Sample Next Token → Decode → Response
                            ↑                    ↓
                            └────────────────────┘
                            (Repeat until done)
```

## 📚 Next Steps

### Improve Model Quality
1. **More training data**: 50K+ samples
2. **Better data quality**: Clean, diverse text
3. **Longer training**: 5-10 epochs
4. **Larger model**: Increase dimensions

### Add Features
1. **System prompts**: Add role/personality
2. **Conversation memory**: Store context
3. **Multi-turn dialogues**: Better context handling
4. **Streaming output**: Show tokens as generated
5. **Web interface**: Build Gradio/Streamlit UI

### Optimize Further
1. **Quantization**: INT8 or INT4 for inference
2. **Knowledge distillation**: Smaller model
3. **LoRA fine-tuning**: Efficient adaptation
4. **Model pruning**: Remove unnecessary weights

## 📖 Resources

### Learning Materials
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer
- [RoFormer](https://arxiv.org/abs/2104.09864) - Rotary Embeddings
- [GQA Paper](https://arxiv.org/abs/2305.13245) - Grouped-Query Attention
- [LLaMA](https://arxiv.org/abs/2302.13971) - Modern LLM architecture

### PyTorch Documentation
- [Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)
- [Distributed Training](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [Model Optimization](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

### HuggingFace
- [Datasets Library](https://huggingface.co/docs/datasets/)
- [Transformers](https://huggingface.co/docs/transformers/)
- [Tokenizers](https://huggingface.co/docs/tokenizers/)

## 🎉 Success Checklist

- [ ] Environment setup complete
- [ ] Test script passes all tests
- [ ] Training completes without errors
- [ ] Model checkpoints saved
- [ ] Chatbot runs and responds
- [ ] Responses are coherent
- [ ] Memory usage acceptable
- [ ] Generation speed acceptable

## 💡 Pro Tips

1. **Start small**: Test with 1000 samples first
2. **Monitor training**: Watch loss curves
3. **Save often**: Don't lose progress
4. **Experiment**: Try different parameters
5. **Be patient**: Quality takes time
6. **Use GPU**: 10-50x faster than CPU
7. **Clean data**: Quality > quantity
8. **Test early**: Catch issues before full training

## 🚨 Important Notes

1. **Windows Users**: Set `num_workers=0` in dataloaders
2. **First Run**: May download tokenizer/models (~500MB)
3. **CUDA Errors**: Update GPU drivers
4. **Memory Leaks**: Call `torch.cuda.empty_cache()` periodically
5. **Checkpoints**: Save to prevent data loss

---

**Congratulations!** 🎉 You now have a complete, production-ready CLI chatbot with efficient training and inference. Enjoy building and experimenting!

For questions or issues, refer to `quick_start.md` or the documentation above.

Happy coding! 🚀
