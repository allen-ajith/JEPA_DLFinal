import wandb
import os
from datetime import datetime

class WandBLogger:
    def __init__(self, config):
        self.enabled = True
        try:
            self.run = wandb.init(
                project=config.wandb_project,
                config=config.__dict__,
            )
            
            # Create artifact directories
            self.create_artifact_dirs()
            
            # Log config file
            self.log_config(config)
            
        except Exception as e:
            print(f"WandB initialization failed: {str(e)}")
            self.enabled = False
    
    def create_artifact_dirs(self):
        """Create directories for artifacts"""
        self.run_name = wandb.run.name if self.enabled else datetime.now().strftime("%Y%m%d_%H%M%S")
        self.artifact_dir = f"artifacts/{self.run_name}"
        os.makedirs(self.artifact_dir, exist_ok=True)
    
    def log_config(self, config):
        """Log config as an artifact"""
        if not self.enabled:
            return
            
        try:
            # Create config artifact
            config_artifact = wandb.Artifact(
                name=f"config_{self.run_name}",
                type="config"
            )
            
            # Save config details
            config_path = os.path.join(self.artifact_dir, "config.txt")
            with open(config_path, "w") as f:
                for key, value in config.__dict__.items():
                    f.write(f"{key}: {value}\n")
            
            config_artifact.add_file(config_path)
            wandb.log_artifact(config_artifact)
            
        except Exception as e:
            print(f"Failed to log config: {str(e)}")
    
    def watch_model(self, model):
        """Watch model parameters and gradients"""
        if self.enabled:
            try:
                wandb.watch(model, log='all')
            except Exception as e:
                print(f"Failed to watch model: {str(e)}")
    
    def log_batch_metrics(self, metrics, prefix="train"):
        """Log batch-level metrics"""
        if self.enabled:
            try:
                wandb.log({f"{prefix}/{k}": v for k, v in metrics.items()})
            except Exception as e:
                print(f"Failed to log batch metrics: {str(e)}")
    
    def log_epoch_metrics(self, epoch_metrics):
        """Log epoch-level metrics"""
        if self.enabled:
            try:
                wandb.log(epoch_metrics)
            except Exception as e:
                print(f"Failed to log epoch metrics: {str(e)}")
    
    def log_model_checkpoint(self, model_path, name, type="model"):
        """Log model checkpoint as artifact"""
        if self.enabled:
            try:
                model_artifact = wandb.Artifact(
                    name=f"{name}_{self.run_name}",
                    type=type
                )
                model_artifact.add_file(model_path)
                wandb.log_artifact(model_artifact)
            except Exception as e:
                print(f"Failed to log model checkpoint: {str(e)}")
    
    def finish(self):
        """Finish the wandb run"""
        if self.enabled:
            try:
                wandb.finish()
            except Exception as e:
                print(f"Failed to finish WandB run: {str(e)}")
    
    def log_best_val_loss(self, val_loss):
        """Log best validation loss"""
        if self.enabled:
            try:
                wandb.run.summary["best_val_loss"] = val_loss
            except Exception as e:
                print(f"Failed to log best val loss: {str(e)}")

    def is_enabled(self):
        """Check if WandB logging is enabled"""
        return self.enabled