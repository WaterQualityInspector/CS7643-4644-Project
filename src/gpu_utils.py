# gpu_utils.py
"""
GPU utility functions for PyTorch models
"""
import torch

def get_device():
    """
    Get the best available device (CUDA if available, otherwise CPU)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")
    
    return device

def move_to_device(obj, device):
    """
    Move tensor/model to device, handling both single tensors and lists/tuples
    """
    if isinstance(obj, (list, tuple)):
        return [move_to_device(item, device) for item in obj]
    elif hasattr(obj, 'to'):
        return obj.to(device)
    else:
        return obj
