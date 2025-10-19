#!/usr/bin/env python3
"""
Convert the trained model checkpoint to .pth format
"""
import sys
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config_from_file
from models.factory import ModelFactory

def main():
    print("🔄 Converting checkpoint to .pth format...")
    
    # Load config
    config_path = "tasks/sigmoid-new-scaling/config.yaml"
    checkpoint_path = "tasks/sigmoid-new-scaling/checkpoints/dit_sigmoid_plateau_best.pt"
    output_path = "tasks/sigmoid-new-scaling/checkpoints/dit_sigmoid_plateau_model.pth"
    
    print(f"📊 Loading config from: {config_path}")
    config = load_config_from_file(config_path)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"🔧 Device: {device}")
    
    # Create model architecture
    print("🏗️  Creating model architecture...")
    model, diffusion = ModelFactory.create_model_and_diffusion(
        config.model,
        config.diffusion,
        device=device
    )
    
    print(f"📦 Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Try to load checkpoint
    print(f"💾 Loading checkpoint from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        print("🔄 Trying alternative loading methods...")
        
        # Try different loading methods
        try:
            # Method 1: Load with weights_only=False
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Model loaded with weights_only=False!")
        except Exception as e2:
            print(f"❌ weights_only=False failed: {e2}")
            
            # Method 2: Try to load just the state dict
            try:
                state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
                if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                    model.load_state_dict(state_dict['model_state_dict'])
                else:
                    model.load_state_dict(state_dict)
                print("✅ Model loaded directly from state dict!")
            except Exception as e3:
                print(f"❌ Direct state dict loading failed: {e3}")
                return
    
    # Save model in .pth format
    print(f"💾 Saving model to: {output_path}")
    model.eval()
    
    # Save the entire model
    torch.save(model, output_path)
    print("✅ Model saved as .pth file!")
    
    # Also save just the state dict
    state_dict_path = "tasks/sigmoid-new-scaling/checkpoints/model_state_dict.pth"
    torch.save(model.state_dict(), state_dict_path)
    print(f"✅ Model state dict saved to: {state_dict_path}")
    
    # Test loading the saved model
    print("🧪 Testing saved model...")
    try:
        loaded_model = torch.load(output_path, map_location=device)
        print("✅ Successfully loaded the saved .pth model!")
        
        # Generate a simple test
        with torch.no_grad():
            test_input = torch.randn(1, 2, 5160, device=device)
            test_geom = torch.randn(1, 3, 5160, device=device)
            test_labels = torch.randn(1, 6, device=device)
            
            # This is just a forward pass test
            output = loaded_model(test_input, test_geom, test_labels)
            print(f"✅ Model forward pass successful! Output shape: {output.shape}")
            
    except Exception as e:
        print(f"❌ Failed to load saved model: {e}")
    
    print("\n🎉 Conversion complete!")
    print(f"📁 Saved files:")
    print(f"   Full model: {output_path}")
    print(f"   State dict: {state_dict_path}")

if __name__ == "__main__":
    main()
