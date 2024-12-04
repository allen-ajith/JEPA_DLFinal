from dataclasses import dataclass
from typing import Optional
from schedulers import LRSchedule

@dataclass
class TrainConfig:
    # Model parameters
    repr_dim: int = 256
    
    # Training parameters
    batch_size: int = 1024
    num_epochs: int = 100
    learning_rate: float = 5e-4
    grad_clip: float = 1.0
    schedule: LRSchedule = LRSchedule.Cosine
    
    # Loss weights
    pred_loss_weight: float = 1.0
    var_loss_weight: float = 0.6
    cov_loss_weight: float = 0.2
    
    # Data paths
    train_data_path: str = "/scratch/DL24FA/train"
    val_data_path: str = "/scratch/DL24FA/probe_normal/val"
    
    # Wandb configs
    wandb_project: str = "jepa-test"
    
    # Device
    device: str = "cuda"
    
    # Save paths
    checkpoint_dir: str = "checkpoints"

# Debug configuration for quick testing
debug_config = TrainConfig()
