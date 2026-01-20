#!/usr/bin/env python3
# Copyright 2025 - Dragon Hatching Training Runner
# Complete training script with monitoring, checkpointing, and evaluation

import os
import sys
import json
import time
import math
import random
import argparse
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from contextlib import nullcontext
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hatching import Hatching, HatchingConfig
from training.data_generators import UnifiedDataGenerator, TaskType
from training.evaluation import UnifiedEvaluator


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: Path, name: str = "training"):
    """Setup logging to file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(log_dir / f"{name}.log")
    fh.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# =============================================================================
# METRICS LOGGER
# =============================================================================

class MetricsLogger:
    """
    Logs training metrics to JSON files for monitoring.

    Creates:
    - metrics.jsonl: Append-only metrics log
    - latest.json: Most recent metrics (for live monitoring)
    - summary.json: Training summary statistics
    """

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.latest_file = self.log_dir / "latest.json"
        self.summary_file = self.log_dir / "summary.json"

        self.history = defaultdict(list)
        self.start_time = time.time()

    def log(self, step: int, metrics: Dict):
        """Log metrics for a step."""
        record = {
            "step": step,
            "timestamp": time.time(),
            "elapsed": time.time() - self.start_time,
            **metrics
        }

        # Append to JSONL
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Update latest
        with open(self.latest_file, "w") as f:
            json.dump(record, f, indent=2)

        # Track history
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not math.isnan(v):
                self.history[k].append(v)

    def get_summary(self) -> Dict:
        """Get summary statistics."""
        summary = {
            "total_time": time.time() - self.start_time,
            "total_steps": len(self.history.get("loss", [])),
        }

        for k, values in self.history.items():
            if values:
                summary[f"{k}_mean"] = np.mean(values[-100:])  # Last 100
                summary[f"{k}_min"] = np.min(values)
                summary[f"{k}_max"] = np.max(values)
                summary[f"{k}_final"] = values[-1]

        return summary

    def save_summary(self):
        """Save summary to file."""
        summary = self.get_summary()
        with open(self.summary_file, "w") as f:
            json.dump(summary, f, indent=2)


# =============================================================================
# NAN CHECKER
# =============================================================================

class NaNChecker:
    """
    Detects NaN/Inf values in tensors and model parameters.

    Usage:
        checker = NaNChecker()
        if checker.check_tensor(loss, "loss"):
            print("NaN detected!")
    """

    def __init__(self, logger=None):
        self.logger = logger
        self.nan_count = 0
        self.inf_count = 0
        self.locations = []

    def check_tensor(self, tensor: torch.Tensor, name: str) -> bool:
        """Check tensor for NaN/Inf. Returns True if bad values found."""
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()

        if has_nan:
            self.nan_count += 1
            self.locations.append(f"NaN in {name}")
            if self.logger:
                self.logger.error(f"🔴 NaN detected in {name}")

        if has_inf:
            self.inf_count += 1
            self.locations.append(f"Inf in {name}")
            if self.logger:
                self.logger.error(f"🔴 Inf detected in {name}")

        return has_nan or has_inf

    def check_model(self, model: nn.Module) -> Dict:
        """Check all model parameters for NaN/Inf."""
        results = {
            "has_nan": False,
            "has_inf": False,
            "nan_params": [],
            "inf_params": [],
        }

        for name, param in model.named_parameters():
            if param is None:
                continue

            if torch.isnan(param).any():
                results["has_nan"] = True
                results["nan_params"].append(name)

            if torch.isinf(param).any():
                results["has_inf"] = True
                results["inf_params"].append(name)

            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    results["has_nan"] = True
                    results["nan_params"].append(f"{name}.grad")

                if torch.isinf(param.grad).any():
                    results["has_inf"] = True
                    results["inf_params"].append(f"{name}.grad")

        return results

    def get_stats(self) -> Dict:
        """Get NaN/Inf statistics."""
        return {
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "locations": self.locations[-10:],  # Last 10
        }


# =============================================================================
# GRADIENT MONITOR
# =============================================================================

class GradientMonitor:
    """
    Monitors gradient statistics for training health.

    Tracks:
    - Gradient norms (should be stable, < 10)
    - Gradient magnitudes
    - Update ratios (|update| / |weight|)
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.history = defaultdict(list)

    def compute_stats(self) -> Dict:
        """Compute gradient statistics."""
        grad_norms = []
        grad_maxs = []
        weight_norms = []

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_max = param.grad.abs().max().item()
                weight_norm = param.norm().item()

                grad_norms.append(grad_norm)
                grad_maxs.append(grad_max)
                weight_norms.append(weight_norm)

        if not grad_norms:
            return {}

        stats = {
            "grad_norm_mean": np.mean(grad_norms),
            "grad_norm_max": np.max(grad_norms),
            "grad_norm_total": np.sqrt(sum(g**2 for g in grad_norms)),
            "grad_max": np.max(grad_maxs),
            "weight_norm_mean": np.mean(weight_norms),
        }

        # Update ratio (how much weights are changing)
        if stats["weight_norm_mean"] > 0:
            stats["update_ratio"] = stats["grad_norm_mean"] / stats["weight_norm_mean"]

        # Track history
        for k, v in stats.items():
            self.history[k].append(v)

        return stats

    def is_healthy(self, stats: Dict) -> Tuple[bool, List[str]]:
        """Check if gradients are healthy."""
        warnings = []

        if stats.get("grad_norm_total", 0) > 100:
            warnings.append(f"Gradient explosion: norm={stats['grad_norm_total']:.2f}")

        if stats.get("grad_norm_total", 1) < 1e-7:
            warnings.append(f"Vanishing gradients: norm={stats['grad_norm_total']:.2e}")

        if stats.get("update_ratio", 0) > 0.1:
            warnings.append(f"Large update ratio: {stats['update_ratio']:.4f}")

        return len(warnings) == 0, warnings


# =============================================================================
# ACTIVATION MONITOR
# =============================================================================

class ActivationMonitor:
    """
    Monitors activation statistics for biological plausibility.

    Tracks:
    - Firing rates (should be ~10% for sparsity)
    - Activation distributions
    """

    def __init__(self):
        self.history = defaultdict(list)

    def compute_stats(self, states: Dict) -> Dict:
        """Compute activation statistics from model states."""
        if not states or "traces" not in states:
            return {}

        stats = {}
        firing_rates = []
        active_fractions = []

        for i, trace in enumerate(states.get("traces", [])):
            if trace is not None:
                # Firing rate from trace
                rate = trace.mean().item()
                firing_rates.append(rate)

                # Active fraction
                active = (trace > 0.1).float().mean().item()
                active_fractions.append(active)

                stats[f"layer_{i}_firing_rate"] = rate
                stats[f"layer_{i}_active_frac"] = active

        if firing_rates:
            stats["firing_rate_mean"] = np.mean(firing_rates)
            stats["firing_rate_std"] = np.std(firing_rates)
            stats["active_fraction_mean"] = np.mean(active_fractions)

        # Track history
        for k, v in stats.items():
            self.history[k].append(v)

        return stats


# =============================================================================
# TRAINING CONFIG
# =============================================================================

@dataclass
class FullTrainingConfig:
    """Complete training configuration."""

    # Run info
    name: str = "hatching_run"
    seed: int = 42

    # Model
    n_layer: int = 6
    n_embd: int = 256
    n_head: int = 4
    dropout: float = 0.1
    vocab_size: int = 256

    # LIF neurons
    tau_mem: float = 10.0
    tau_syn: float = 5.0
    v_threshold: float = 1.0

    # STDP
    tau_plus: float = 20.0
    tau_minus: float = 20.0
    A_plus: float = 0.01
    A_minus: float = 0.0105

    # Training
    batch_size: int = 32
    max_seq_len: int = 512
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    weight_decay: float = 0.1
    max_iters: int = 10000
    warmup_iters: int = 500
    grad_clip: float = 1.0

    # Biological regularization
    sparsity_target: float = 0.1
    sparsity_weight: float = 0.001

    # Curriculum
    curriculum_enabled: bool = True

    # Logging
    log_freq: int = 100
    eval_freq: int = 500
    save_freq: int = 1000
    log_dir: str = "logs"

    # Device
    device: str = "auto"
    dtype: str = "auto"
    compile: bool = True

    # Safety
    nan_check: bool = True
    nan_threshold: int = 3  # Stop after this many NaNs

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "FullTrainingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# TRAINER
# =============================================================================

class DragonTrainer:
    """
    Complete training pipeline for Dragon Hatching.

    Features:
    - Multi-task curriculum learning
    - Comprehensive monitoring
    - NaN detection and recovery
    - Automatic checkpointing
    - Evaluation during training
    """

    def __init__(self, config: FullTrainingConfig):
        self.config = config

        # Setup
        self._setup_seed()
        self._setup_device()
        self._setup_logging()
        self._setup_model()
        self._setup_optimizer()
        self._setup_data()
        self._setup_monitors()

        self.global_step = 0
        self.best_loss = float("inf")

    def _setup_seed(self):
        """Set random seeds for reproducibility."""
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _setup_device(self):
        """Setup compute device and dtype."""
        # Device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        # Dtype
        if self.config.dtype == "auto":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.dtype = "bfloat16"
            else:
                self.dtype = "float32"
        else:
            self.dtype = self.config.dtype

        self.ptdtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[self.dtype]

        # Autocast context
        self.ctx = (
            torch.amp.autocast(device_type=self.device.type, dtype=self.ptdtype)
            if "cuda" in str(self.device)
            else nullcontext()
        )

        self.scaler = torch.amp.GradScaler(
            device=self.device.type,
            enabled=(self.dtype == "float16")
        )

        self.logger.info(f"Device: {self.device}, dtype: {self.dtype}")

    def _setup_logging(self):
        """Setup logging infrastructure."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(self.config.log_dir) / f"{self.config.name}_{timestamp}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logging(self.log_dir, "training")
        self.metrics_logger = MetricsLogger(self.log_dir)

        # Save config
        with open(self.log_dir / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        self.logger.info(f"Logging to: {self.log_dir}")

    def _setup_model(self):
        """Initialize model."""
        hatching_config = HatchingConfig(
            n_layer=self.config.n_layer,
            n_embd=self.config.n_embd,
            n_head=self.config.n_head,
            dropout=self.config.dropout,
            vocab_size=self.config.vocab_size,
            tau_mem=self.config.tau_mem,
            tau_syn=self.config.tau_syn,
            v_threshold=self.config.v_threshold,
            tau_plus=self.config.tau_plus,
            tau_minus=self.config.tau_minus,
            A_plus=self.config.A_plus,
            A_minus=self.config.A_minus,
        )

        self.model = Hatching(hatching_config).to(self.device)

        # Compile if requested and available
        if self.config.compile:
            try:
                self.model = torch.compile(self.model)
                self.logger.info("Model compiled with torch.compile()")
            except Exception as e:
                self.logger.warning(f"torch.compile() failed: {e}")

        n_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model parameters: {n_params:,}")

    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95),
        )

        # Cosine scheduler with warmup
        def lr_lambda(step):
            if step < self.config.warmup_iters:
                return step / max(1, self.config.warmup_iters)

            progress = (step - self.config.warmup_iters) / max(
                1, self.config.max_iters - self.config.warmup_iters
            )
            return self.config.min_lr / self.config.learning_rate + (
                1 - self.config.min_lr / self.config.learning_rate
            ) * (1 + math.cos(math.pi * progress)) / 2

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )

    def _setup_data(self):
        """Setup data generators."""
        self.data_gen = UnifiedDataGenerator()

        # Curriculum stages
        self.curriculum_stages = [
            {"iters": 2000, "weights": {
                TaskType.TEXT_COMPLETION: 0.7,
                TaskType.CHAIN_OF_THOUGHT: 0.3
            }},
            {"iters": 3000, "weights": {
                TaskType.TEXT_COMPLETION: 0.3,
                TaskType.FUNCTION_CALL: 0.4,
                TaskType.CHAIN_OF_THOUGHT: 0.3
            }},
            {"iters": 3000, "weights": {
                TaskType.TEXT_COMPLETION: 0.2,
                TaskType.FUNCTION_CALL: 0.3,
                TaskType.BASH_COMMAND: 0.3,
                TaskType.CHAIN_OF_THOUGHT: 0.2
            }},
            {"iters": 2000, "weights": {
                TaskType.TEXT_COMPLETION: 0.15,
                TaskType.FUNCTION_CALL: 0.2,
                TaskType.BASH_COMMAND: 0.2,
                TaskType.TOOL_USE: 0.25,
                TaskType.CHAIN_OF_THOUGHT: 0.2
            }},
        ]

        self.current_stage = 0

    def _setup_monitors(self):
        """Setup monitoring tools."""
        self.nan_checker = NaNChecker(self.logger)
        self.grad_monitor = GradientMonitor(self.model)
        self.activation_monitor = ActivationMonitor()
        self.evaluator = UnifiedEvaluator()

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a training batch."""
        # Update curriculum stage
        if self.config.curriculum_enabled:
            cumulative = 0
            for i, stage in enumerate(self.curriculum_stages):
                cumulative += stage["iters"]
                if self.global_step < cumulative:
                    if i != self.current_stage:
                        self.current_stage = i
                        self.logger.info(f">>> Curriculum Stage {i+1}")
                        self.data_gen.weights = stage["weights"]
                    break

        # Generate batch
        samples = self.data_gen.generate_batch(self.config.batch_size)

        # Convert to tensors
        batch_x = []
        batch_y = []

        for sample in samples:
            text = sample.input_text + sample.target_text
            tokens = list(text.encode("utf-8"))

            # Truncate/pad
            if len(tokens) > self.config.max_seq_len:
                tokens = tokens[:self.config.max_seq_len]
            else:
                tokens = tokens + [0] * (self.config.max_seq_len - len(tokens))

            batch_x.append(tokens[:-1])
            batch_y.append(tokens[1:])

        x = torch.tensor(batch_x, dtype=torch.long, device=self.device)
        y = torch.tensor(batch_y, dtype=torch.long, device=self.device)

        return x, y

    def train_step(self) -> Dict:
        """Single training step."""
        self.model.train()

        # Get batch
        x, y = self.get_batch()

        # Forward pass
        with self.ctx:
            logits, loss, states = self.model(x, y, return_states=True)

            # Sparsity regularization
            sparsity_loss = torch.tensor(0.0, device=self.device)
            if states and "traces" in states:
                for trace in states["traces"]:
                    if trace is not None:
                        rate = trace.mean()
                        sparsity_loss = sparsity_loss + (
                            rate - self.config.sparsity_target
                        ) ** 2

            total_loss = loss + self.config.sparsity_weight * sparsity_loss

        # NaN check
        if self.config.nan_check:
            if self.nan_checker.check_tensor(total_loss, "loss"):
                return {"loss": float("nan"), "nan_detected": True}

        # Backward pass
        self.scaler.scale(total_loss).backward()

        # Gradient clipping
        if self.config.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            )

        # Gradient stats
        grad_stats = self.grad_monitor.compute_stats()

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.scheduler.step()

        # Activation stats
        activation_stats = self.activation_monitor.compute_stats(states)

        return {
            "loss": loss.item(),
            "sparsity_loss": sparsity_loss.item(),
            "lr": self.scheduler.get_last_lr()[0],
            "stage": self.current_stage + 1,
            **grad_stats,
            **activation_stats,
        }

    @torch.no_grad()
    def evaluate(self, n_samples: int = 20) -> Dict:
        """Evaluate model."""
        self.model.eval()

        total_loss = 0
        coherence_scores = []
        function_scores = []
        bash_scores = []

        for _ in range(n_samples):
            x, y = self.get_batch()
            x = x[:4]  # Smaller batch for eval
            y = y[:4]

            logits, loss, _ = self.model(x, y)
            total_loss += loss.item()

            # Generate and evaluate
            prompt = x[0:1, :32]
            generated = self.model.generate(prompt, max_new_tokens=64, temperature=0.8)
            text = bytes(generated[0].cpu().tolist()).decode(errors="replace")

            eval_result = self.evaluator.evaluate(text)
            coherence_scores.append(eval_result.coherence.overall_score)
            if eval_result.function_call.overall_score > 0:
                function_scores.append(eval_result.function_call.overall_score)
            if eval_result.bash.overall_score > 0:
                bash_scores.append(eval_result.bash.overall_score)

        self.model.train()

        return {
            "val_loss": total_loss / n_samples,
            "coherence": np.mean(coherence_scores) if coherence_scores else 0,
            "function_acc": np.mean(function_scores) if function_scores else 0,
            "bash_acc": np.mean(bash_scores) if bash_scores else 0,
        }

    def save_checkpoint(self, name: str = None):
        """Save checkpoint."""
        name = name or f"hatching_step_{self.global_step}.pt"
        path = self.log_dir / name

        checkpoint = {
            "step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "config": self.config.to_dict(),
            "best_loss": self.best_loss,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
            }
        }

        if torch.cuda.is_available():
            checkpoint["rng_state"]["cuda"] = torch.cuda.get_rng_state_all()

        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint and return step."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.best_loss = checkpoint.get("best_loss", float("inf"))

        # Restore RNG state
        rng_state = checkpoint.get("rng_state", {})
        if "python" in rng_state:
            random.setstate(rng_state["python"])
        if "numpy" in rng_state:
            np.random.set_state(rng_state["numpy"])
        if "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        if "cuda" in rng_state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state["cuda"])

        self.global_step = checkpoint["step"]
        self.logger.info(f"Loaded checkpoint from step {self.global_step}")

        return self.global_step

    def train(self, resume_from: str = None):
        """Main training loop."""
        if resume_from:
            self.load_checkpoint(resume_from)

        self.logger.info("=" * 60)
        self.logger.info("DRAGON HATCHING TRAINING")
        self.logger.info("=" * 60)

        loss_acc = 0
        loss_count = 0
        start_time = time.time()

        try:
            for step in range(self.global_step, self.config.max_iters):
                self.global_step = step

                # Training step
                metrics = self.train_step()

                # Check for NaN
                if metrics.get("nan_detected"):
                    self.nan_checker.nan_count += 1
                    if self.nan_checker.nan_count >= self.config.nan_threshold:
                        self.logger.error("Too many NaNs, stopping training")
                        break
                    continue

                loss_acc += metrics["loss"]
                loss_count += 1

                # Logging
                if step % self.config.log_freq == 0 and step > 0:
                    avg_loss = loss_acc / loss_count
                    elapsed = time.time() - start_time
                    iters_per_sec = step / elapsed if elapsed > 0 else 0

                    log_msg = (
                        f"Step {step:5d}/{self.config.max_iters} | "
                        f"Loss: {avg_loss:.4f} | "
                        f"LR: {metrics['lr']:.2e} | "
                        f"Stage: {metrics['stage']} | "
                        f"Speed: {iters_per_sec:.1f} it/s"
                    )

                    if "firing_rate_mean" in metrics:
                        log_msg += f" | FR: {metrics['firing_rate_mean']:.3f}"

                    self.logger.info(log_msg)

                    # Log metrics
                    self.metrics_logger.log(step, {
                        "loss": avg_loss,
                        **metrics
                    })

                    loss_acc = 0
                    loss_count = 0

                # Evaluation
                if step % self.config.eval_freq == 0 and step > 0:
                    eval_metrics = self.evaluate()
                    self.logger.info(
                        f"  Eval | val_loss: {eval_metrics['val_loss']:.4f} | "
                        f"coherence: {eval_metrics['coherence']:.3f} | "
                        f"func: {eval_metrics['function_acc']:.3f} | "
                        f"bash: {eval_metrics['bash_acc']:.3f}"
                    )

                    self.metrics_logger.log(step, {f"eval_{k}": v for k, v in eval_metrics.items()})

                    # Save best
                    if eval_metrics["val_loss"] < self.best_loss:
                        self.best_loss = eval_metrics["val_loss"]
                        self.save_checkpoint("hatching_best.pt")

                # Checkpointing
                if step % self.config.save_freq == 0 and step > 0:
                    self.save_checkpoint()

        except KeyboardInterrupt:
            self.logger.info("Training interrupted")

        # Final save
        self.save_checkpoint("hatching_final.pt")
        self.metrics_logger.save_summary()

        self.logger.info("=" * 60)
        self.logger.info("Training complete!")
        self.logger.info(f"Best loss: {self.best_loss:.4f}")
        self.logger.info(f"Logs saved to: {self.log_dir}")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train Dragon Hatching model")

    # Basic
    parser.add_argument("--name", type=str, default="hatching", help="Run name")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Config file (JSON)")

    # Model
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--n-head", type=int, default=4)

    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Logging
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--eval-freq", type=int, default=500)
    parser.add_argument("--save-freq", type=int, default=1000)

    # Device
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-compile", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config from file if provided
    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
        config = FullTrainingConfig.from_dict(config_dict)
    else:
        config = FullTrainingConfig(
            name=args.name,
            n_layer=args.n_layer,
            n_embd=args.n_embd,
            n_head=args.n_head,
            batch_size=args.batch_size,
            max_iters=args.max_iters,
            learning_rate=args.lr,
            grad_clip=args.grad_clip,
            log_dir=args.log_dir,
            log_freq=args.log_freq,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq,
            device=args.device,
            compile=not args.no_compile,
        )

    # Train
    trainer = DragonTrainer(config)
    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
