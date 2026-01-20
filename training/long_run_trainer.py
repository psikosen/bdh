# Copyright 2025 - Long-Running Training Infrastructure for Dragon Hatching
#
# Features:
# - Automatic checkpointing with configurable frequency
# - Graceful interruption handling (Ctrl+C saves checkpoint)
# - Learning rate scheduling with warmup
# - Gradient accumulation for large effective batch sizes
# - Mixed precision training (AMP)
# - Real-time metrics logging (TensorBoard, W&B, JSON)
# - Automatic resume from latest checkpoint
# - Curriculum learning support
# - Early stopping with patience

import json
import logging
import math
import os
import signal
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm
import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Complete training configuration."""

    # Model
    model_type: str = "hatching"  # "bdh", "hatching", "advanced"
    n_layer: int = 6
    n_embd: int = 256
    n_head: int = 4
    vocab_size: int = 50257  # GPT-2 vocab
    dropout: float = 0.1
    mlp_multiplier: int = 128

    # Biological parameters (for Hatching)
    tau_mem: float = 10.0
    tau_syn: float = 5.0
    tau_plus: float = 20.0
    v_threshold: float = 1.0

    # Training
    max_steps: int = 100000
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_seq_len: int = 512

    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0

    # Learning rate schedule
    warmup_steps: int = 1000
    lr_decay_steps: int = 100000
    min_lr_ratio: float = 0.1

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    checkpoint_freq: int = 1000
    keep_last_n_checkpoints: int = 5

    # Logging
    log_dir: str = "./logs"
    log_freq: int = 10
    eval_freq: int = 500

    # Hardware
    device: str = "auto"
    dtype: str = "bfloat16"  # "float32", "float16", "bfloat16"
    compile_model: bool = True

    # Curriculum learning
    curriculum_stages: List[Dict] = field(default_factory=list)

    # Early stopping
    early_stopping_patience: int = 0  # 0 = disabled
    early_stopping_min_delta: float = 0.001

    # Experiment tracking
    experiment_name: str = "bdh_training"
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "bdh"

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str):
        """Save config to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def get_effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


# =============================================================================
# METRICS TRACKER
# =============================================================================

class MetricsTracker:
    """Track and log training metrics."""

    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "bdh",
        config: Optional[Dict] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb

        # JSON log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_path = self.log_dir / f"{experiment_name}_{timestamp}.jsonl"

        # TensorBoard
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.log_dir / "tensorboard" / f"{experiment_name}_{timestamp}"
                self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
                logger.info(f"TensorBoard logging to {tb_dir}")
            except ImportError:
                logger.warning("TensorBoard not available")

        # Weights & Biases
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=f"{experiment_name}_{timestamp}",
                    config=config,
                )
                logger.info(f"W&B logging to {wandb_project}/{experiment_name}")
            except ImportError:
                logger.warning("W&B not available")

        # In-memory history for plotting
        self.history: Dict[str, List[Tuple[int, float]]] = {}

    def log(self, step: int, metrics: Dict[str, float]):
        """Log metrics to all backends."""
        # Add timestamp
        metrics_with_meta = {
            "step": step,
            "timestamp": time.time(),
            **metrics,
        }

        # JSON
        with open(self.json_path, 'a') as f:
            f.write(json.dumps(metrics_with_meta) + '\n')

        # TensorBoard
        if self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)

        # W&B
        if self.wandb_run:
            import wandb
            wandb.log(metrics, step=step)

        # In-memory
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append((step, value))

    def log_histogram(self, step: int, name: str, values: torch.Tensor):
        """Log histogram to TensorBoard."""
        if self.tb_writer:
            self.tb_writer.add_histogram(name, values, step)

    def log_text(self, step: int, name: str, text: str):
        """Log text to TensorBoard."""
        if self.tb_writer:
            self.tb_writer.add_text(name, text, step)

    def close(self):
        """Close all logging backends."""
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            import wandb
            wandb.finish()


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================

class CheckpointManager:
    """Manage model checkpoints with automatic cleanup."""

    def __init__(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 5,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        config: TrainingConfig,
        metrics: Dict[str, float],
        extra: Optional[Dict] = None,
    ) -> str:
        """Save checkpoint and return path."""
        checkpoint = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "config": asdict(config),
            "metrics": metrics,
            "extra": extra or {},
            "timestamp": datetime.now().isoformat(),
        }

        path = self.checkpoint_dir / f"checkpoint_step_{step:08d}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

        # Cleanup old checkpoints
        self._cleanup()

        # Save "latest" symlink
        latest_path = self.checkpoint_dir / "latest.pt"
        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        latest_path.symlink_to(path.name)

        return str(path)

    def load_latest(self) -> Optional[Dict]:
        """Load the most recent checkpoint."""
        latest_path = self.checkpoint_dir / "latest.pt"
        if latest_path.exists():
            return self.load(str(latest_path))

        # Fallback: find newest checkpoint
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if checkpoints:
            return self.load(str(checkpoints[-1]))

        return None

    def load(self, path: str) -> Dict:
        """Load a specific checkpoint."""
        logger.info(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return checkpoint

    def _cleanup(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_step_*.pt"))

        if len(checkpoints) > self.keep_last_n:
            for old_ckpt in checkpoints[:-self.keep_last_n]:
                old_ckpt.unlink()
                logger.debug(f"Removed old checkpoint: {old_ckpt}")


# =============================================================================
# LEARNING RATE SCHEDULER
# =============================================================================

def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    decay_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Create learning rate scheduler with warmup and cosine decay.

    Schedule:
    - Linear warmup from 0 to max_lr over warmup_steps
    - Cosine decay from max_lr to min_lr over remaining steps
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup
            return step / max(1, warmup_steps)
        elif step < decay_steps:
            # Cosine decay
            progress = (step - warmup_steps) / max(1, decay_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            # Minimum learning rate
            return min_lr_ratio

    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# TRAINER
# =============================================================================

class LongRunTrainer:
    """
    Long-running trainer with all bells and whistles.

    Features:
    - Automatic checkpointing
    - Graceful interruption (Ctrl+C)
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Real-time metrics
    - Early stopping
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        config: TrainingConfig,
        eval_dataset: Optional[Dataset] = None,
    ):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # Device setup
        if config.device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(config.device)

        logger.info(f"Using device: {self.device}")

        # Dtype setup
        self.dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(config.dtype, torch.float32)

        # Move model to device
        self.model = self.model.to(self.device)

        # Compile model if requested (PyTorch 2.0+)
        if config.compile_model and hasattr(torch, 'compile'):
            logger.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model)

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
        )

        # Scheduler
        self.scheduler = get_lr_scheduler(
            self.optimizer,
            config.warmup_steps,
            config.lr_decay_steps,
            config.min_lr_ratio,
        )

        # Mixed precision
        self.scaler = torch.amp.GradScaler(
            device=self.device.type,
            enabled=(config.dtype != "float32"),
        )

        # Data loader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,  # For simplicity
            pin_memory=(self.device.type == "cuda"),
        )

        if eval_dataset:
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_size=config.batch_size,
                shuffle=False,
            )
        else:
            self.eval_loader = None

        # Checkpointing
        self.checkpoint_manager = CheckpointManager(
            config.checkpoint_dir,
            config.keep_last_n_checkpoints,
        )

        # Metrics
        self.metrics = MetricsTracker(
            config.log_dir,
            config.experiment_name,
            config.use_tensorboard,
            config.use_wandb,
            config.wandb_project,
            asdict(config),
        )

        # State
        self.global_step = 0
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
        self.should_stop = False

        # Signal handling for graceful interruption
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Set up handlers for graceful interruption."""
        def signal_handler(signum, frame):
            logger.info("\nInterrupted! Saving checkpoint before exit...")
            self._save_checkpoint(interrupt=True)
            self.should_stop = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _save_checkpoint(self, interrupt: bool = False):
        """Save current training state."""
        metrics = {
            "best_eval_loss": self.best_eval_loss,
            "patience_counter": self.patience_counter,
        }

        extra = {
            "interrupted": interrupt,
            "scaler_state_dict": self.scaler.state_dict(),
        }

        self.checkpoint_manager.save(
            self.global_step,
            self.model,
            self.optimizer,
            self.scheduler,
            self.config,
            metrics,
            extra,
        )

    def resume_from_checkpoint(self) -> bool:
        """Resume training from latest checkpoint. Returns True if resumed."""
        checkpoint = self.checkpoint_manager.load_latest()

        if checkpoint is None:
            logger.info("No checkpoint found, starting fresh")
            return False

        self.global_step = checkpoint["step"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "scaler_state_dict" in checkpoint.get("extra", {}):
            self.scaler.load_state_dict(checkpoint["extra"]["scaler_state_dict"])

        self.best_eval_loss = checkpoint["metrics"].get("best_eval_loss", float('inf'))
        self.patience_counter = checkpoint["metrics"].get("patience_counter", 0)

        logger.info(f"Resumed from step {self.global_step}")
        return True

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """Execute single training step."""
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        # Forward pass with mixed precision
        with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
            logits, loss, states = self.model(x, y)

        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps

        # Backward pass
        self.scaler.scale(loss).backward()

        metrics = {
            "train/loss": loss.item() * self.config.gradient_accumulation_steps,
        }

        # Add biological metrics if available
        if states and isinstance(states, dict):
            if "firing_rate" in states:
                metrics["train/firing_rate"] = states["firing_rate"]
            if "sparsity" in states:
                metrics["train/sparsity"] = states["sparsity"]

        return metrics

    def optimizer_step(self):
        """Execute optimizer step with gradient clipping."""
        # Unscale gradients for clipping
        self.scaler.unscale_(self.optimizer)

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        # Scheduler step
        self.scheduler.step()

        return {"train/grad_norm": grad_norm.item()}

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        if self.eval_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for batch in self.eval_loader:
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.amp.autocast(device_type=self.device.type, dtype=self.dtype):
                logits, loss, _ = self.model(x, y)

            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

        self.model.train()

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 20))  # Cap to avoid overflow

        return {
            "eval/loss": avg_loss,
            "eval/perplexity": perplexity,
        }

    def check_early_stopping(self, eval_loss: float) -> bool:
        """Check if training should stop early."""
        if self.config.early_stopping_patience <= 0:
            return False

        if eval_loss < self.best_eval_loss - self.config.early_stopping_min_delta:
            self.best_eval_loss = eval_loss
            self.patience_counter = 0
            return False

        self.patience_counter += 1

        if self.patience_counter >= self.config.early_stopping_patience:
            logger.info(f"Early stopping triggered after {self.patience_counter} evaluations without improvement")
            return True

        return False

    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.config.max_steps} steps")
        logger.info(f"Effective batch size: {self.config.get_effective_batch_size()}")

        self.model.train()
        data_iter = iter(self.train_loader)

        # Progress bar
        pbar = tqdm(
            initial=self.global_step,
            total=self.config.max_steps,
            desc="Training",
            unit="step",
        )

        accumulation_metrics = {}

        try:
            while self.global_step < self.config.max_steps and not self.should_stop:
                # Get batch (with infinite data loader)
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)

                # Training step
                step_metrics = self.train_step(batch)

                # Accumulate metrics
                for k, v in step_metrics.items():
                    accumulation_metrics[k] = accumulation_metrics.get(k, 0) + v

                # Gradient accumulation
                if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    opt_metrics = self.optimizer_step()

                    # Average accumulated metrics
                    for k in list(accumulation_metrics.keys()):
                        accumulation_metrics[k] /= self.config.gradient_accumulation_steps

                    accumulation_metrics.update(opt_metrics)
                    accumulation_metrics["train/lr"] = self.scheduler.get_last_lr()[0]

                self.global_step += 1
                pbar.update(1)

                # Logging
                if self.global_step % self.config.log_freq == 0:
                    self.metrics.log(self.global_step, accumulation_metrics)

                    # Update progress bar
                    loss_str = f"loss={accumulation_metrics.get('train/loss', 0):.4f}"
                    lr_str = f"lr={accumulation_metrics.get('train/lr', 0):.2e}"
                    pbar.set_postfix_str(f"{loss_str} {lr_str}")

                    accumulation_metrics = {}

                # Evaluation
                if self.global_step % self.config.eval_freq == 0:
                    eval_metrics = self.evaluate()
                    if eval_metrics:
                        self.metrics.log(self.global_step, eval_metrics)
                        logger.info(f"Step {self.global_step}: eval_loss={eval_metrics['eval/loss']:.4f}")

                        # Early stopping check
                        if self.check_early_stopping(eval_metrics['eval/loss']):
                            self.should_stop = True

                # Checkpointing
                if self.global_step % self.config.checkpoint_freq == 0:
                    self._save_checkpoint()

        finally:
            pbar.close()

            # Final checkpoint
            if not self.should_stop:
                self._save_checkpoint()

            self.metrics.close()
            logger.info(f"Training finished at step {self.global_step}")


# =============================================================================
# SIMPLE DATASET FOR TESTING
# =============================================================================

class TextDataset(Dataset):
    """Simple text dataset for training."""

    def __init__(self, data: torch.Tensor, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return max(1, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="BDH Long-Run Training")
    parser.add_argument("--config", type=str, help="Path to config YAML")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--steps", type=int, help="Override max_steps")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training.log"),
        ],
    )

    # Load config
    if args.config:
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig()

    # Override from command line
    if args.steps:
        config.max_steps = args.steps
    if args.lr:
        config.learning_rate = args.lr

    # Create model
    if config.model_type == "hatching":
        from hatching import Hatching, HatchingConfig
        model_config = HatchingConfig(
            n_layer=config.n_layer,
            n_embd=config.n_embd,
            n_head=config.n_head,
            vocab_size=config.vocab_size,
            dropout=config.dropout,
            mlp_internal_dim_multiplier=config.mlp_multiplier,
            tau_mem=config.tau_mem,
            tau_syn=config.tau_syn,
            tau_plus=config.tau_plus,
            v_threshold=config.v_threshold,
        )
        model = Hatching(model_config)
    else:
        from bdh import BDH, BDHConfig
        model_config = BDHConfig(
            n_layer=config.n_layer,
            n_embd=config.n_embd,
            n_head=config.n_head,
            vocab_size=config.vocab_size,
            dropout=config.dropout,
            mlp_internal_dim_multiplier=config.mlp_multiplier,
        )
        model = BDH(model_config)

    logger.info(f"Model: {config.model_type} with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create dummy dataset (replace with real data)
    logger.info("Creating dataset...")
    data = torch.randint(0, config.vocab_size, (100000,))
    train_data = data[:90000]
    eval_data = data[90000:]

    train_dataset = TextDataset(train_data, config.max_seq_len)
    eval_dataset = TextDataset(eval_data, config.max_seq_len)

    # Create trainer
    trainer = LongRunTrainer(model, train_dataset, config, eval_dataset)

    # Resume if requested
    if args.resume:
        trainer.resume_from_checkpoint()

    # Train!
    trainer.train()


if __name__ == "__main__":
    main()
