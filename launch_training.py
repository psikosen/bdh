#!/usr/bin/env python3
# Copyright 2025 - BDH Training Launcher
#
# Unified launcher for training with:
# - Configuration management
# - Automatic monitoring
# - GPU detection and optimization
# - Experiment tracking
# - Easy resume from checkpoints

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION UTILITIES
# =============================================================================

def load_config(config_path: str) -> dict:
    """Load and validate configuration."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Set defaults
    defaults = {
        'model_type': 'hatching',
        'n_layer': 6,
        'n_embd': 256,
        'n_head': 4,
        'vocab_size': 50257,
        'dropout': 0.1,
        'mlp_multiplier': 128,
        'tau_mem': 10.0,
        'tau_syn': 5.0,
        'tau_plus': 20.0,
        'v_threshold': 1.0,
        'max_steps': 100000,
        'batch_size': 32,
        'gradient_accumulation_steps': 1,
        'max_seq_len': 512,
        'learning_rate': 3e-4,
        'weight_decay': 0.1,
        'beta1': 0.9,
        'beta2': 0.95,
        'max_grad_norm': 1.0,
        'warmup_steps': 1000,
        'lr_decay_steps': 100000,
        'min_lr_ratio': 0.1,
        'checkpoint_dir': './checkpoints',
        'checkpoint_freq': 1000,
        'keep_last_n_checkpoints': 5,
        'log_dir': './logs',
        'log_freq': 10,
        'eval_freq': 500,
        'device': 'auto',
        'dtype': 'bfloat16',
        'compile_model': True,
        'experiment_name': 'bdh_training',
        'use_tensorboard': True,
        'use_wandb': False,
        'wandb_project': 'bdh',
        'early_stopping_patience': 0,
        'early_stopping_min_delta': 0.001,
        'curriculum_stages': [],
    }

    for key, value in defaults.items():
        if key not in config:
            config[key] = value

    return config


def detect_hardware() -> dict:
    """Detect available hardware and optimal settings."""
    info = {
        'device': 'cpu',
        'dtype': 'float32',
        'gpu_name': None,
        'gpu_memory': None,
        'cuda_available': False,
        'mps_available': False,
    }

    try:
        import torch

        if torch.cuda.is_available():
            info['cuda_available'] = True
            info['device'] = 'cuda'
            info['gpu_name'] = torch.cuda.get_device_name(0)
            info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory / 1e9

            # Check for bfloat16 support
            if torch.cuda.is_bf16_supported():
                info['dtype'] = 'bfloat16'
            else:
                info['dtype'] = 'float16'

        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info['mps_available'] = True
            info['device'] = 'mps'
            info['dtype'] = 'float32'  # MPS doesn't support bfloat16 well

    except ImportError:
        pass

    return info


def recommend_batch_size(gpu_memory_gb: float, model_size: str = 'medium') -> int:
    """Recommend batch size based on GPU memory."""
    if gpu_memory_gb is None:
        return 8  # Conservative CPU default

    # Rough estimates based on model size
    base_memory_per_sample = {
        'small': 0.1,   # ~10M params
        'medium': 0.3,  # ~100M params
        'large': 1.0,   # ~1B params
    }.get(model_size, 0.3)

    # Leave some memory for activations and overhead
    usable_memory = gpu_memory_gb * 0.7
    recommended = int(usable_memory / base_memory_per_sample)

    # Clamp to reasonable range
    return max(4, min(128, recommended))


# =============================================================================
# LAUNCHER
# =============================================================================

def print_banner():
    """Print startup banner."""
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   🐉  BABY DRAGON HATCHLING - Training Launcher                              ║
║                                                                               ║
║   Biologically-inspired neural network training                               ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_config(config: dict, hardware: dict):
    """Print configuration summary."""
    print("\n📋 Configuration:")
    print(f"   Model: {config['model_type']} ({config['n_layer']}L, {config['n_embd']}D, {config['n_head']}H)")
    print(f"   Training: {config['max_steps']:,} steps, batch_size={config['batch_size']}")
    print(f"   Learning Rate: {config['learning_rate']:.2e} (warmup={config['warmup_steps']})")

    print(f"\n🖥️  Hardware:")
    print(f"   Device: {hardware['device']}")
    if hardware['gpu_name']:
        print(f"   GPU: {hardware['gpu_name']} ({hardware['gpu_memory']:.1f} GB)")
    print(f"   Dtype: {config['dtype']}")
    print(f"   Compile: {config['compile_model']}")

    print(f"\n📁 Paths:")
    print(f"   Checkpoints: {config['checkpoint_dir']}")
    print(f"   Logs: {config['log_dir']}")
    print(f"   Experiment: {config['experiment_name']}")


def launch_training(config: dict, resume: bool = False):
    """Launch the training process."""
    # Import here to avoid startup delay
    sys.path.insert(0, str(Path(__file__).parent))

    from training.long_run_trainer import (
        TrainingConfig,
        LongRunTrainer,
        TextDataset,
    )

    # Convert dict to TrainingConfig
    training_config = TrainingConfig(**config)

    # Create model
    logger.info("Creating model...")

    if config['model_type'] == 'hatching':
        from hatching import Hatching, HatchingConfig
        model_config = HatchingConfig(
            n_layer=config['n_layer'],
            n_embd=config['n_embd'],
            n_head=config['n_head'],
            vocab_size=config['vocab_size'],
            dropout=config['dropout'],
            mlp_internal_dim_multiplier=config['mlp_multiplier'],
            tau_mem=config['tau_mem'],
            tau_syn=config['tau_syn'],
            tau_plus=config['tau_plus'],
            v_threshold=config['v_threshold'],
        )
        model = Hatching(model_config)
    else:
        from bdh import BDH, BDHConfig
        model_config = BDHConfig(
            n_layer=config['n_layer'],
            n_embd=config['n_embd'],
            n_head=config['n_head'],
            vocab_size=config['vocab_size'],
            dropout=config['dropout'],
            mlp_internal_dim_multiplier=config['mlp_multiplier'],
        )
        model = BDH(model_config)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model has {n_params:,} parameters")

    # Create dataset (replace with real data loading)
    logger.info("Loading dataset...")
    import torch

    # Check for existing data file
    data_path = Path("input.txt")
    if data_path.exists():
        logger.info(f"Loading data from {data_path}")
        with open(data_path, 'r') as f:
            text = f.read()
        data = torch.tensor([ord(c) for c in text], dtype=torch.long)
        # Remap to vocab_size if needed
        data = data % config['vocab_size']
    else:
        logger.info("No input.txt found, using random data for demo")
        data = torch.randint(0, config['vocab_size'], (500000,))

    # Split
    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]

    train_dataset = TextDataset(train_data, config['max_seq_len'])
    eval_dataset = TextDataset(eval_data, config['max_seq_len'])

    logger.info(f"Train samples: {len(train_dataset):,}")
    logger.info(f"Eval samples: {len(eval_dataset):,}")

    # Create trainer
    trainer = LongRunTrainer(model, train_dataset, training_config, eval_dataset)

    # Resume if requested
    if resume:
        if trainer.resume_from_checkpoint():
            logger.info("Resumed from checkpoint")
        else:
            logger.info("No checkpoint found, starting fresh")

    # Train!
    print("\n" + "=" * 70)
    print("🚀 Starting training...")
    print("   Press Ctrl+C to gracefully stop and save checkpoint")
    print("=" * 70 + "\n")

    trainer.train()

    print("\n✅ Training complete!")


def launch_monitor(config: dict):
    """Launch the live monitor in a separate process."""
    monitor_cmd = [
        sys.executable,
        str(Path(__file__).parent / "training" / "live_monitor.py"),
        "--log-dir", config['log_dir'],
        "--max-steps", str(config['max_steps']),
    ]

    logger.info("Launching monitor...")
    return subprocess.Popen(monitor_cmd)


def launch_tensorboard(config: dict):
    """Launch TensorBoard in background."""
    tb_dir = Path(config['log_dir']) / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)

    tb_cmd = [
        sys.executable, "-m", "tensorboard.main",
        "--logdir", str(tb_dir),
        "--port", "6006",
        "--bind_all",
    ]

    logger.info("Launching TensorBoard on http://localhost:6006")
    return subprocess.Popen(tb_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BDH Training Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start training with default config
  python launch_training.py

  # Start with custom config
  python launch_training.py --config configs/my_experiment.yaml

  # Resume from checkpoint
  python launch_training.py --resume

  # Start with live monitor
  python launch_training.py --monitor

  # Quick test run
  python launch_training.py --quick
        """
    )

    parser.add_argument("--config", "-c", type=str, default="configs/default.yaml",
                       help="Path to configuration YAML file")
    parser.add_argument("--resume", "-r", action="store_true",
                       help="Resume from latest checkpoint")
    parser.add_argument("--monitor", "-m", action="store_true",
                       help="Launch live monitor alongside training")
    parser.add_argument("--tensorboard", "-t", action="store_true",
                       help="Launch TensorBoard")
    parser.add_argument("--quick", "-q", action="store_true",
                       help="Quick test run (1000 steps)")

    # Override options
    parser.add_argument("--steps", type=int, help="Override max_steps")
    parser.add_argument("--lr", type=float, help="Override learning_rate")
    parser.add_argument("--batch-size", type=int, help="Override batch_size")
    parser.add_argument("--name", type=str, help="Override experiment_name")

    args = parser.parse_args()

    print_banner()

    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        logger.info(f"Loading config from {config_path}")
        config = load_config(str(config_path))
    else:
        logger.info(f"Config not found at {config_path}, using defaults")
        config = load_config(str(Path(__file__).parent / "configs" / "default.yaml"))

    # Apply overrides
    if args.steps:
        config['max_steps'] = args.steps
    if args.lr:
        config['learning_rate'] = args.lr
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.name:
        config['experiment_name'] = args.name
    if args.quick:
        config['max_steps'] = 1000
        config['eval_freq'] = 100
        config['checkpoint_freq'] = 500
        config['log_freq'] = 10

    # Detect hardware
    hardware = detect_hardware()

    # Auto-configure device and dtype if set to 'auto'
    if config['device'] == 'auto':
        config['device'] = hardware['device']
    if config['dtype'] == 'auto':
        config['dtype'] = hardware['dtype']

    # Print summary
    print_config(config, hardware)

    # Create directories
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['log_dir']).mkdir(parents=True, exist_ok=True)

    # Save effective config
    effective_config_path = Path(config['log_dir']) / f"{config['experiment_name']}_config.yaml"
    with open(effective_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved effective config to {effective_config_path}")

    # Launch TensorBoard if requested
    tb_process = None
    if args.tensorboard and config['use_tensorboard']:
        tb_process = launch_tensorboard(config)
        time.sleep(2)  # Give it time to start

    # Launch monitor if requested
    monitor_process = None
    if args.monitor:
        monitor_process = launch_monitor(config)
        time.sleep(1)

    # Launch training
    try:
        launch_training(config, resume=args.resume)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        # Cleanup
        if monitor_process:
            monitor_process.terminate()
        if tb_process:
            tb_process.terminate()


if __name__ == "__main__":
    main()
