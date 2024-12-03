import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn.functional as F
from models import SimpleJEPAModel
from dataset import create_wall_dataloader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys

def compute_jepa_loss(model, states, actions):
    """JEPA loss as VICReg with normalized weights"""
    with torch.no_grad():
        target_embeddings = model.get_target_embeddings(states)
    
    predicted_embeddings = model(states, actions)
    
    # 1. Prediction Loss aka Invariance Loss
    pred_loss = F.mse_loss(predicted_embeddings[:, 1:], target_embeddings[:, 1:])
    
    # 2. Variance Loss - prevent collapse
    variances = predicted_embeddings.var(dim=0)
    var_loss = torch.mean(torch.relu(1.0 - variances))
    
    # 3. Covariance Loss - following VICReg paper
    embedding_dim = predicted_embeddings.size(-1)  # D = 256
    pred_flat = predicted_embeddings.reshape(-1, embedding_dim)
    pred_flat = pred_flat - pred_flat.mean(dim=0)
    
    N = pred_flat.size(0)
    cov = (pred_flat.t() @ pred_flat) / (N - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = off_diag.pow(2).sum() / embedding_dim
    
    # Normalized scaling weights (paper original 25 : 25 : 1)  
    total_loss = 1.0 * pred_loss + 0.8 * var_loss + 0.2 * cov_loss
    
    return total_loss, {
        'total_loss': total_loss.item(),
        'raw_pred_loss': pred_loss.item(),
        'raw_var_loss': var_loss.item(),
        'raw_cov_loss': cov_loss.item()
    }

def validate(model, val_loader):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            loss, _ = compute_jepa_loss(model, batch.states, batch.actions)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

BATCH_SIZE = 512

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleJEPAModel().to(device)
    
    train_loader = create_wall_dataloader(
        data_path="/scratch/DL24FA/train",
        probing=False,
        device=device,
        batch_size=BATCH_SIZE,
        train=True
    )
    
    val_loader = create_wall_dataloader(
        data_path="/scratch/DL24FA/probe_normal/val",
        probing=True,
        device=device,
        batch_size=BATCH_SIZE,
        train=False
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    #scheduler = CosineAnnealingLR(optimizer, T_max=50)
    
    num_epochs = 100
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, leave=False, 
                   desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            loss, loss_components = compute_jepa_loss(model, batch.states, batch.actions)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss_components['total_loss']
            num_batches += 1
            
            # Update progress bar with current stats
            pbar.set_postfix({
                'total': f"{loss_components['total_loss']:.4f}",
                'pred': f"{loss_components['raw_pred_loss']:.4f}",
                'var': f"{loss_components['raw_var_loss']:.4f}",
                'cov': f"{loss_components['raw_cov_loss']:.4f}"
            })
            if batch_idx % 65 == 0:
                # Clear the current line
                pbar.clear()
                # Print stats
                print(f"Epoch {epoch} | Batch {batch_idx:5d} | Total: {loss_components['total_loss']:10.4f} | "
                      f"Pred: {loss_components['raw_pred_loss']:7.4f} | "
                      f"Var: {loss_components['raw_var_loss']:7.4f} | "
                      f"Cov: {loss_components['raw_cov_loss']:7.4f}")
                # Force progress bar to refresh
                pbar.refresh()
        
        val_loss = validate(model, val_loader)
        
        avg_train_loss = total_loss / num_batches
        print(f"\nEpoch {epoch} Summary:")
        print(f"Average Train Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}\n")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
        
        torch.save(model.state_dict(), "latest_model.pth")
        #scheduler.step()

if __name__ == "__main__":
    train()