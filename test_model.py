"""
Quick test script to verify the model architecture works correctly.
Run this before training to catch any issues early.
"""
import torch
from LLM_architecture_168 import CogniMamba, ModelArgs, load_config

def test_model():
    """Test basic model functionality."""
    print("="*60)
    print("Testing Cogni-Mamba Model Architecture")
    print("="*60)
    
    # Load config
    print("\n1. Loading configuration...")
    try:
        config = load_config('config.json')
        model_args = ModelArgs(**config)
        print("   ✓ Configuration loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading config: {e}")
        return False
    
    # Create model
    print("\n2. Creating model...")
    try:
        model = CogniMamba(model_args)
        num_params = model.get_num_params()
        print(f"   ✓ Model created with {num_params/1e6:.2f}M parameters")
    except Exception as e:
        print(f"   ✗ Error creating model: {e}")
        return False
    
    # Test forward pass (CPU)
    print("\n3. Testing forward pass (CPU)...")
    try:
        batch_size = 2
        seq_len = 64
        dummy_input = torch.randint(0, model_args.vocab_size, (batch_size, seq_len))
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        expected_shape = (batch_size, seq_len, model_args.vocab_size)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        print(f"   ✓ Forward pass successful")
        print(f"   ✓ Output shape: {output.shape}")
    except Exception as e:
        print(f"   ✗ Error in forward pass: {e}")
        return False
    
    # Test GPU if available
    if torch.cuda.is_available():
        print("\n4. Testing forward pass (GPU)...")
        try:
            model = model.to('cuda')
            dummy_input = dummy_input.to('cuda')
            
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"   ✓ GPU forward pass successful")
            print(f"   ✓ GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        except Exception as e:
            print(f"   ✗ Error in GPU forward pass: {e}")
            return False
    else:
        print("\n4. GPU not available, skipping GPU test")
    
    # Test gradient flow
    print("\n5. Testing gradient flow...")
    try:
        model = model.cpu()  # Move back to CPU for testing
        model.train()
        
        dummy_input = torch.randint(0, model_args.vocab_size, (batch_size, seq_len))
        output = model(dummy_input)
        
        # Create dummy target and compute loss
        target = torch.randint(0, model_args.vocab_size, (batch_size, seq_len))
        loss = torch.nn.functional.cross_entropy(
            output.view(-1, model_args.vocab_size),
            target.view(-1)
        )
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, "No gradients computed"
        
        print(f"   ✓ Gradient flow successful")
        print(f"   ✓ Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"   ✗ Error in gradient flow: {e}")
        return False
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
    print("\nYou can now proceed with training:")
    print("  python train.py")
    print()
    
    return True


if __name__ == '__main__':
    success = test_model()
    exit(0 if success else 1)
