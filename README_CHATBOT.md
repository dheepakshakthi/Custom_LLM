# Simple Chatbot with SmolLM2-135M

A simple, interactive chatbot built using the SmolLM2-135M model from HuggingFace.

## Features

- 🤖 Interactive command-line chatbot
- 💬 Maintains conversation history for context
- 🎛️ Configurable generation parameters
- 🔄 Clear conversation history command
- 💻 Works on both CPU and GPU
- 📝 Programmatic API for integration

## Requirements

The required packages should already be installed in your virtual environment:
- transformers
- torch
- (Optional) cuda for GPU support

## Usage

### Interactive Mode (Default)

Simply run the chatbot to start an interactive conversation:

```bash
python chatbot.py
```

Or use the example script:

```bash
python chatbot_example.py
```

### Commands in Interactive Mode

- Type your message to chat with the bot
- Type `clear` to clear conversation history
- Type `quit`, `exit`, or `bye` to end the conversation

### Programmatic Mode

Run a pre-programmed conversation:

```bash
python chatbot_example.py --programmatic
```

### Using in Your Own Code

```python
from chatbot import SimpleChatbot

# Create chatbot instance
bot = SimpleChatbot(device="cpu")  # Use "cuda" for GPU

# Single message
response = bot.chat("Hello, how are you?")
print(response)

# Continue conversation (history is maintained)
response = bot.chat("What's the weather like?")
print(response)

# Clear history if needed
bot.clear_history()
```

## Configuration

You can customize the chatbot behavior:

```python
# Initialize with specific device
bot = SimpleChatbot(device="cuda")  # or "cpu"

# Generate with custom parameters
response = bot.generate_response(
    user_input="Your question",
    max_length=150,      # Maximum response length
    temperature=0.8,     # Higher = more random (0.1-1.0)
    top_p=0.9           # Nucleus sampling parameter
)
```

## Model Information

- **Model**: SmolLM2-135M
- **Source**: HuggingFaceTB/SmolLM2-135M
- **Size**: 135M parameters
- **Type**: Causal Language Model

## Tips

1. **GPU vs CPU**: The chatbot auto-detects GPU availability. For faster responses, use GPU if available.

2. **Response Quality**: This is a small model (135M parameters), so responses may be:
   - Short and simple
   - Sometimes inconsistent
   - Best for demonstration purposes

3. **Conversation Context**: The bot remembers the last 6 messages (3 exchanges) for context.

4. **Generation Parameters**:
   - Lower `temperature` (0.3-0.5) = more focused, deterministic responses
   - Higher `temperature` (0.7-1.0) = more creative, random responses

## Example Session

```
You: Hello! What's your name?
🤖 Bot: I'm a helpful AI assistant.

You: What is Python?
🤖 Bot: Python is a high-level programming language.

You: clear
🗑️  Conversation history cleared.

You: quit
👋 Goodbye! Thanks for chatting!
```

## Troubleshooting

**Issue**: Model downloads slowly
- The model is downloaded from HuggingFace on first run
- Subsequent runs will use the cached model

**Issue**: Out of memory
- Try using CPU instead of GPU: `SimpleChatbot(device="cpu")`
- Reduce `max_length` parameter

**Issue**: Responses seem random
- Lower the `temperature` parameter
- Try temperature=0.5 for more consistent responses

## License

This chatbot uses the SmolLM2-135M model. Please refer to the model's license on HuggingFace for usage terms.
