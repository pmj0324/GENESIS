
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # Try to load with different numpy versions
    import numpy as np
    print(f'Current numpy version: {np.__version__}')
    
    # Load checkpoint
    checkpoint_path = 'tasks/sigmoid-new-scaling/checkpoints/dit_sigmoid_plateau_best.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f'Checkpoint keys: {list(checkpoint.keys())}')
    
    # Save only the model state dict
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # Save as a simple state dict
    torch.save(model_state_dict, 'tasks/sigmoid-new-scaling/checkpoints/model_state_dict.pth')
    print('✅ Saved model state dict as .pth file')
    
except Exception as e:
    print(f'Error: {e}')
