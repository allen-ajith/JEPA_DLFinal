import torch
import torch.nn.functional as F

def compute_jepa_loss(model, states, actions, config):
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
    embedding_dim = predicted_embeddings.size(-1)
    pred_flat = predicted_embeddings.reshape(-1, embedding_dim)
    pred_flat = pred_flat - pred_flat.mean(dim=0)
    
    N = pred_flat.size(0)
    cov = (pred_flat.t() @ pred_flat) / (N - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = off_diag.pow(2).sum() / embedding_dim
    
    # Apply weights from config
    total_loss = (
        config.pred_loss_weight * pred_loss + 
        config.var_loss_weight * var_loss + 
        config.cov_loss_weight * cov_loss
    )
    
    return total_loss, {
        'total_loss': total_loss.item(),
        'pred_loss': pred_loss.item(),
        'var_loss': var_loss.item(),
        'cov_loss': cov_loss.item()
    }