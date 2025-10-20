# ✅ Getting Started Checklist

## Before You Begin

- [ ] Python 3.8+ installed (`python --version`)
- [ ] CUDA GPU available (optional but recommended)
- [ ] 16GB+ RAM available
- [ ] 50GB+ free disk space
- [ ] BookCorpus dataset in `archive/` folder (or willing to use dummy data)

## Installation Steps

- [ ] Navigate to project directory
- [ ] Create virtual environment: `python -m venv master`
- [ ] Activate environment: `.\master\Scripts\activate`
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Verify installation: `python quickstart.py` → option 7

## Understanding the Project

- [ ] Read `README.md` for comprehensive overview
- [ ] Read `SETUP.md` for setup instructions
- [ ] Review `PROJECT_SUMMARY.md` for technical details
- [ ] Check `config.py` to see all settings

## Training Preparation

- [ ] Check GPU availability: `nvidia-smi` (Windows) or `torch.cuda.is_available()`
- [ ] Verify dataset location: `archive/BookCorpus3.csv` exists
- [ ] Review training config in `config.py`
- [ ] Ensure enough disk space for checkpoints

## Training Process

- [ ] Start training: `python train.py`
- [ ] Monitor progress (loss, perplexity shown in terminal)
- [ ] Wait for pre-training phase to complete (~4-12 hours)
- [ ] Wait for fine-tuning phase to complete (~2-6 hours)
- [ ] Check for completion message
- [ ] Verify outputs created:
  - [ ] `checkpoints/` folder exists
  - [ ] `training_stats/` folder exists
  - [ ] `training_outputs/llm_200m_final.pt` exists

## After Training

- [ ] Visualize results: `python visualize_metrics.py`
- [ ] Review training metrics in `training_stats/training_metrics.json`
- [ ] Check for overfitting in plots
- [ ] Review summary statistics

## Using the Chatbot

- [ ] Start chatbot: `python chatbot.py`
- [ ] Type `help` to see commands
- [ ] Test with simple questions
- [ ] Try adjusting temperature: `temp 0.9`
- [ ] Try adjusting length: `length 200`
- [ ] Clear history: `clear`
- [ ] Exit when done: `exit`

## Testing Different Checkpoints

- [ ] Try pre-trained checkpoint: `python chatbot.py --checkpoint pretrain`
- [ ] Try fine-tuned checkpoint: `python chatbot.py --checkpoint finetune`
- [ ] Compare responses between checkpoints
- [ ] Note which performs better

## Troubleshooting

If you encounter issues:

- [ ] CUDA out of memory?
  - Edit `config.py`: reduce `batch_size` to 4 or 2
  
- [ ] Dataset not loading?
  - Code will use dummy data automatically
  - Continue with training for testing
  
- [ ] Training too slow?
  - Verify GPU is being used
  - Check `nvidia-smi` for GPU utilization
  - Consider reducing model size in `config.py`
  
- [ ] Module not found?
  - Ensure virtual environment is activated
  - Run `pip install -r requirements.txt` again
  
- [ ] Chatbot not responding well?
  - Try different checkpoints
  - Adjust temperature (0.7-1.0 recommended)
  - Use fine-tuned checkpoint for best results

## Optional: Advanced Usage

- [ ] Modify model architecture in `model.py`
- [ ] Adjust hyperparameters in `config.py`
- [ ] Add custom datasets in `data_processing.py`
- [ ] Implement custom generation strategies
- [ ] Export model for deployment

## Best Practices

- [ ] Always activate virtual environment before running scripts
- [ ] Keep checkpoints backed up (they're valuable!)
- [ ] Monitor GPU memory during training
- [ ] Review statistics after each training run
- [ ] Experiment with generation parameters
- [ ] Document your experiments

## Quick Commands Reference

```powershell
# Activate environment
.\master\Scripts\activate

# Train
python train.py

# Chat
python chatbot.py

# Visualize
python visualize_metrics.py

# Quick menu
python quickstart.py

# Check config
python config.py
```

## Success Criteria

You'll know everything is working when:

✅ Training completes without errors
✅ Loss decreases over time
✅ Checkpoints are saved
✅ Chatbot loads and responds
✅ Responses are coherent (after fine-tuning)
✅ Statistics are saved and visualized

## Need Help?

1. Check error messages carefully
2. Review relevant documentation:
   - `README.md` - Full docs
   - `SETUP.md` - Setup guide
   - `PROJECT_SUMMARY.md` - Technical details
3. Check code comments
4. Use `quickstart.py` for guided operations

---

## Final Checklist

Before considering the project complete:

- [ ] Training finished successfully
- [ ] All checkpoints saved
- [ ] Statistics visualized
- [ ] Chatbot tested and working
- [ ] Responses are reasonable quality
- [ ] All files understood
- [ ] Can explain the architecture
- [ ] Can adjust hyperparameters
- [ ] Can use different checkpoints

**Congratulations! You've built a 200M parameter LLM! 🎉**

---

**Estimated Time to Complete:**
- Setup: 30 minutes
- Training: 6-18 hours (depending on hardware)
- Testing: 30 minutes
- **Total: ~7-19 hours**

**Good luck with your training! 🚀**
