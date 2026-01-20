# BDH - Baby Dragon Hatchling
# Makefile for common development tasks

.PHONY: help install install-dev train train-quick monitor tensorboard visualize test clean

# Default target
help:
	@echo "BDH - Baby Dragon Hatchling"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Setup:"
	@echo "  install       Install core dependencies"
	@echo "  install-dev   Install all dependencies (including dev tools)"
	@echo "  install-all   Install everything"
	@echo ""
	@echo "Training:"
	@echo "  train         Start training with default config"
	@echo "  train-quick   Quick test run (1000 steps)"
	@echo "  train-resume  Resume training from checkpoint"
	@echo ""
	@echo "Monitoring:"
	@echo "  monitor       Launch live training monitor"
	@echo "  tensorboard   Launch TensorBoard dashboard"
	@echo ""
	@echo "Visualization:"
	@echo "  visualize     Generate training visualizations"
	@echo ""
	@echo "Development:"
	@echo "  test          Run test suite"
	@echo "  lint          Run linter"
	@echo "  format        Format code with black"
	@echo "  clean         Remove generated files"

# =============================================================================
# SETUP
# =============================================================================

install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

# =============================================================================
# TRAINING
# =============================================================================

train:
	python launch_training.py --config configs/default.yaml

train-quick:
	python launch_training.py --quick

train-resume:
	python launch_training.py --resume

train-monitor:
	python launch_training.py --monitor --tensorboard

# Custom training with arguments
# Usage: make train-custom ARGS="--steps 50000 --lr 1e-4"
train-custom:
	python launch_training.py $(ARGS)

# =============================================================================
# MONITORING
# =============================================================================

monitor:
	python training/live_monitor.py --log-dir ./logs

tensorboard:
	tensorboard --logdir ./logs/tensorboard --port 6006

# =============================================================================
# VISUALIZATION
# =============================================================================

visualize:
	python training/visualize.py ./logs/*.jsonl --output ./figures --compare

visualize-interactive:
	python training/visualize.py ./logs/*.jsonl --interactive

# =============================================================================
# DEVELOPMENT
# =============================================================================

test:
	pytest tests/ -v

test-math:
	pytest tests/test_math_proofs.py tests/test_mathematical_invariants.py -v

lint:
	@echo "Running linters..."
	python -m py_compile bdh.py hatching.py hatching_advanced.py
	@echo "Syntax OK"

format:
	black --line-length 100 *.py training/*.py tests/*.py

# =============================================================================
# CLEANUP
# =============================================================================

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -rf build dist
	rm -rf figures/*.png figures/*.pdf

clean-checkpoints:
	rm -rf checkpoints/*

clean-logs:
	rm -rf logs/*

clean-all: clean clean-checkpoints clean-logs

# =============================================================================
# DATA
# =============================================================================

download-data:
	@echo "Downloading Tiny Shakespeare..."
	curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
	@echo "Done! Data saved to input.txt"

# =============================================================================
# DOCKER (optional)
# =============================================================================

docker-build:
	docker build -t bdh:latest .

docker-train:
	docker run --gpus all -v $(PWD):/workspace bdh:latest python launch_training.py

# =============================================================================
# UTILITIES
# =============================================================================

count-params:
	@python -c "from hatching import Hatching, HatchingConfig; m = Hatching(HatchingConfig()); print(f'Parameters: {sum(p.numel() for p in m.parameters()):,}')"

gpu-info:
	@python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
