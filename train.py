import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn.functional as F
from models import SimpleJEPAModel
from dataset import create_wall_dataloader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from trainconfig import TrainConfig, debug_config
from losses import compute_jepa_loss
from wandblogger import WandBLogger

def validate(model, val_loader, config, logger=None):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            loss, metrics = compute_jepa_loss(model, batch.states, batch.actions, config)
            total_loss += metrics['total_loss']
            num_batches += 1
            
            if logger and logger.is_enabled():
                logger.log_batch_metrics(metrics, prefix="val")
    
    return total_loss / num_batches

def save_checkpoint(model, optimizer, epoch, val_loss, config, is_best=False):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }
    
    # Save latest checkpoint
    latest_path = os.path.join(config.checkpoint_dir, 'latest_model.pth')
    torch.save(checkpoint, latest_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        return best_path
    return latest_path

def train(config: TrainConfig):
    # Initialize WandB logger
    logger = WandBLogger(config)
    
    # Create model
    model = SimpleJEPAModel(repr_dim=config.repr_dim).to(config.device)
    logger.watch_model(model)
    
    # Create data loaders
    train_loader = create_wall_dataloader(
        data_path=config.train_data_path,
        probing=False,
        device=config.device,
        batch_size=config.batch_size,
        train=True
    )
    
    val_loader = create_wall_dataloader(
        data_path=config.val_data_path,
        probing=True,
        device=config.device,
        batch_size=config.batch_size,
        train=False
    )
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, leave=False, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            loss, metrics = compute_jepa_loss(model, batch.states, batch.actions, config)
            
            optimizer.zero_grad()
            loss.backward()
            
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    config.grad_clip
                )
            
            optimizer.step()
            
            total_loss += metrics['total_loss']
            num_batches += 1
            
            logger.log_batch_metrics(metrics)
            
            pbar.set_postfix({
                'total': f"{metrics['total_loss']:.4f}",
                'pred': f"{metrics['pred_loss']:.4f}",
                'var': f"{metrics['var_loss']:.4f}",
                'cov': f"{metrics['cov_loss']:.4f}"
            })
            
            if batch_idx % 130 == 0:
                pbar.clear()
                print(f"Epoch {epoch} | Batch {batch_idx:5d} | "
                      f"Total: {metrics['total_loss']:10.4f} | "
                      f"Pred: {metrics['pred_loss']:7.4f} | "
                      f"Var: {metrics['var_loss']:7.4f} | "
                      f"Cov: {metrics['cov_loss']:7.4f}")
                pbar.refresh()
        
        val_loss = validate(model, val_loader, config, logger)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log epoch metrics
        logger.log_epoch_metrics({
            "epoch": epoch,
            "train/epoch_loss": total_loss / num_batches,
            "val/epoch_loss": val_loss,
            "learning_rate": current_lr
        })
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"Average Train Loss: {total_loss / num_batches:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Learning Rate: {current_lr:.6f}\n")
        
        # Save checkpoints
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            logger.log_best_val_loss(best_val_loss)
        
        # Save and log checkpoints
        checkpoint_path = save_checkpoint(
            model, optimizer, epoch, val_loss,
            config, is_best
        )
        logger.log_model_checkpoint(
            checkpoint_path,
            f"model_epoch_{epoch}"
        )
    
    logger.finish()

if __name__ == "__main__":
    selected_config = debug_config
    train(config=selected_config)