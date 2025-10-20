# 🚀 Quick Setup Guide for LLM-200M

## Step-by-Step Setup

### 1. Verify Python Installation
```powershell
python --version
# Should show Python 3.8 or higher
```

### 2. Create and Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv master

# Activate it
.\master\Scripts\activate

# Your prompt should now show (master)
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

This will install:
- PyTorch (deep learning framework)
- Transformers (HuggingFace library)
- Datasets (data loading)
- Pandas, NumPy (data processing)
- Colorama (colored terminal output)
- Matplotlib (visualization)
- And more...

### 4. Verify Installation
```powershell
python quickstart.py
# Choose option 7 to test model
# Choose option 8 to test data loading
```

### 5. Start Training
```powershell
python train.py
```

**Important Notes:**
- Training will take several hours (GPU recommended)
- Requires ~16GB RAM and ~50GB disk space
- If BookCorpus dataset fails to load, the code will use dummy data for testing
- Training statistics are saved automatically in `training_stats/`
- Checkpoints are saved in `checkpoints/`

### 6. Monitor Training Progress

The training script shows real-time progress:
- Loss and perplexity metrics
- Learning rate schedule
- Progress bars with ETA
- Validation metrics (during fine-tuning)

### 7. Use the Chatbot

After training completes:

```powershell
# Use the final model
python chatbot.py

# Or use a specific checkpoint
python chatbot.py --checkpoint finetune
python chatbot.py --checkpoint pretrain
```

### 8. Visualize Training Results

```powershell
python visualize_metrics.py
```

This will:
- Print training summary
- Generate plots of loss, perplexity, and learning rate
- Save plots to `training_stats/training_plots.png`

---

## Alternative: Use Quick Start Menu

For easier navigation:

```powershell
python quickstart.py
```

This provides an interactive menu for all common tasks.

---

## Troubleshooting

### Issue: CUDA out of memory
**Solution:** Edit `train.py` and reduce `batch_size` from 8 to 4 or 2

### Issue: Dataset not loading
**Solution:** The code will automatically use dummy data for testing. You can still train and test the model.

### Issue: Training is very slow
**Solution:** 
- Make sure you're using GPU (check with `nvidia-smi` in PowerShell)
- Reduce dataset size for testing
- Reduce model size by editing `config.py`

### Issue: Module not found
**Solution:** Make sure virtual environment is activated and dependencies are installed:
```powershell
.\master\Scripts\activate
pip install -r requirements.txt
```

---

## Quick Commands Reference

```powershell
# Activate environment
.\master\Scripts\activate

# Train model
python train.py

# Chat with model
python chatbot.py

# Visualize training
python visualize_metrics.py

# Quick start menu
python quickstart.py

# View configuration
python config.py

# Deactivate environment
deactivate
```

---

## What Gets Created During Training?

```
Custom_LLM/
├── cache/                      # Cached tokenized data
│   └── bookcorpus_tokenized.pt
├── checkpoints/                # Model checkpoints
│   ├── pretrain_best.pt
│   ├── pretrain_latest.pt
│   ├── finetune_best.pt
│   └── finetune_latest.pt
├── training_stats/             # Training metrics
│   ├── training_metrics.json
│   └── training_plots.png
└── training_outputs/           # Final model
    └── llm_200m_final.pt
```

---

## Expected Training Time

| Hardware | Pre-training | Fine-tuning | Total |
|----------|--------------|-------------|-------|
| RTX 3090 | ~4-6 hours  | ~2-3 hours  | ~6-9 hours |
| RTX 3080 | ~6-8 hours  | ~3-4 hours  | ~9-12 hours |
| RTX 3070 | ~8-12 hours | ~4-6 hours  | ~12-18 hours |
| CPU only | Several days | Several days | Not recommended |

---

## Tips for Best Results

1. **Use GPU**: Training on CPU is extremely slow
2. **Monitor GPU memory**: Use `nvidia-smi` to check usage
3. **Save checkpoints**: Training saves automatically, don't worry about interruptions
4. **Check statistics**: Use `visualize_metrics.py` to monitor progress
5. **Experiment**: Try different generation parameters in chatbot for better responses

---

## Need Help?

1. Check the README.md for detailed documentation
2. Look at the code comments for explanations
3. Use the quickstart.py menu for guided operations
4. Check training_stats/ for metrics and plots

---

**Ready to start? Run:**
```powershell
.\master\Scripts\activate
python quickstart.py
```

**Happy training! 🚀**
