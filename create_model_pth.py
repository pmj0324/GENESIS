#!/usr/bin/env python3
"""
Create a .pth model file from the training configuration
"""
import sys
import os
import warnings
import torch
import torch.nn as nn
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class DiTModel(nn.Module):
    """DiT (Diffusion Transformer) model based on the training configuration"""
    
    def __init__(self, seq_len=5160, hidden=256, depth=3, heads=8, dropout=0.1, 
                 label_dim=6, t_embed_dim=128, mlp_ratio=2.0):
        super().__init__()
        
        self.seq_len = seq_len
        self.hidden = hidden
        self.depth = depth
        self.heads = heads
        self.label_dim = label_dim
        
        # Input projection
        self.input_proj = nn.Linear(2, hidden)  # charge, time
        
        # Geometry projection
        self.geom_proj = nn.Linear(3, hidden)  # x, y, z
        
        # Label embedding
        self.label_embed = nn.Linear(label_dim, hidden)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden, t_embed_dim),
            nn.SiLU(),
            nn.Linear(t_embed_dim, hidden)
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(hidden, heads, dropout, mlp_ratio)
            for _ in range(depth)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden, 2)  # charge, time
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden)
        
    def forward(self, x, geom, labels, timesteps):
        """
        Forward pass
        x: (batch, 2, seq_len) - charge and time
        geom: (batch, 3, seq_len) - x, y, z coordinates
        labels: (batch, 6) - event labels
        timesteps: (batch,) - diffusion timesteps
        """
        batch_size = x.size(0)
        
        # Project inputs
        x_proj = self.input_proj(x.transpose(1, 2))  # (batch, seq_len, hidden)
        geom_proj = self.geom_proj(geom.transpose(1, 2))  # (batch, seq_len, hidden)
        
        # Add geometry information
        x_proj = x_proj + geom_proj
        
        # Label embedding
        label_emb = self.label_embed(labels)  # (batch, hidden)
        
        # Time embedding
        time_emb = self.time_embed(label_emb)  # (batch, hidden)
        
        # Add time embedding to each position
        x_proj = x_proj + time_emb.unsqueeze(1)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x_proj = block(x_proj)
        
        # Final normalization and projection
        x_proj = self.norm(x_proj)
        output = self.output_proj(x_proj)  # (batch, seq_len, 2)
        
        return output.transpose(1, 2)  # (batch, 2, seq_len)

class TransformerBlock(nn.Module):
    """Transformer block with self-attention and MLP"""
    
    def __init__(self, hidden, heads, dropout, mlp_ratio):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        
        # MLP
        mlp_hidden = int(hidden * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # MLP
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        
        return x

def create_model_pth():
    """Create and save the model in .pth format"""
    print("🏗️  Creating DiT model based on training configuration...")
    
    # Model configuration from training
    model_config = {
        'seq_len': 5160,
        'hidden': 256,
        'depth': 3,
        'heads': 8,
        'dropout': 0.1,
        'label_dim': 6,
        't_embed_dim': 128,
        'mlp_ratio': 2.0
    }
    
    # Create model
    model = DiTModel(**model_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📦 Model created:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    model.apply(init_weights)
    
    # Set to evaluation mode
    model.eval()
    
    # Save model
    output_dir = Path("tasks/sigmoid-new-scaling/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full model
    model_path = output_dir / "dit_sigmoid_plateau_model.pth"
    torch.save(model, model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # Save state dict
    state_dict_path = output_dir / "dit_sigmoid_plateau_state_dict.pth"
    torch.save(model.state_dict(), state_dict_path)
    print(f"✅ State dict saved to: {state_dict_path}")
    
    # Test the model
    print("🧪 Testing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create test inputs
    batch_size = 2
    x = torch.randn(batch_size, 2, 5160, device=device)
    geom = torch.randn(batch_size, 3, 5160, device=device)
    labels = torch.randn(batch_size, 6, device=device)
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    
    with torch.no_grad():
        output = model(x, geom, labels, timesteps)
    
    print(f"✅ Model test successful!")
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    
    # Save model info
    info_path = output_dir / "model_info.txt"
    with open(info_path, 'w') as f:
        f.write("DiT Model Information\n")
        f.write("=====================\n\n")
        f.write(f"Architecture: DiT (Diffusion Transformer)\n")
        f.write(f"Sequence length: {model_config['seq_len']}\n")
        f.write(f"Hidden dimension: {model_config['hidden']}\n")
        f.write(f"Depth (layers): {model_config['depth']}\n")
        f.write(f"Attention heads: {model_config['heads']}\n")
        f.write(f"Dropout: {model_config['dropout']}\n")
        f.write(f"Label dimension: {model_config['label_dim']}\n")
        f.write(f"Time embed dimension: {model_config['t_embed_dim']}\n")
        f.write(f"MLP ratio: {model_config['mlp_ratio']}\n\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")
        f.write(f"Model size: {total_params * 4 / 1024 / 1024:.1f} MB\n")
        f.write(f"\nInput shapes:\n")
        f.write(f"  x (signal): (batch, 2, {model_config['seq_len']})\n")
        f.write(f"  geom: (batch, 3, {model_config['seq_len']})\n")
        f.write(f"  labels: (batch, 6)\n")
        f.write(f"  timesteps: (batch,)\n")
        f.write(f"\nOutput shape:\n")
        f.write(f"  output: (batch, 2, {model_config['seq_len']})\n")
    
    print(f"✅ Model info saved to: {info_path}")
    
    return model_path, state_dict_path, info_path

def main():
    print("🚀 Creating .pth model file...")
    print("="*70)
    
    try:
        model_path, state_dict_path, info_path = create_model_pth()
        
        print("\n" + "="*70)
        print("✅ Model creation complete!")
        print("="*70)
        print(f"\n📁 Files created:")
        print(f"   Full model: {model_path}")
        print(f"   State dict: {state_dict_path}")
        print(f"   Model info: {info_path}")
        
        print(f"\n🔍 Usage:")
        print(f"   # Load full model")
        print(f"   model = torch.load('{model_path}')")
        print(f"   ")
        print(f"   # Load state dict")
        print(f"   model = DiTModel()")
        print(f"   model.load_state_dict(torch.load('{state_dict_path}'))")
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
