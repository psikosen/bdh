#!/usr/bin/env python3
# Copyright 2025 - Visualization Utilities for Dragon Hatching Training
#
# Features:
# - Training curves (loss, perplexity, learning rate)
# - Biological metrics visualization (firing rates, sparsity)
# - Gradient flow analysis
# - Weight distribution histograms
# - Attention pattern visualization
# - Spike raster plots
# - Comparative analysis across runs
# - Publication-quality figures

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Plotting imports (with fallbacks)
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MaxNLocator
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# =============================================================================
# DATA LOADING
# =============================================================================

@dataclass
class TrainingRun:
    """Container for training run data."""
    name: str
    steps: List[int]
    train_loss: List[float]
    eval_loss: List[float]
    learning_rate: List[float]
    grad_norm: List[float]
    firing_rate: List[float]
    sparsity: List[float]
    timestamps: List[float]

    @classmethod
    def from_jsonl(cls, path: str, name: Optional[str] = None) -> "TrainingRun":
        """Load training data from JSONL log file."""
        if name is None:
            name = Path(path).stem

        steps = []
        train_loss = []
        eval_loss = []
        learning_rate = []
        grad_norm = []
        firing_rate = []
        sparsity = []
        timestamps = []

        with open(path) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())

                    step = entry.get('step', 0)
                    steps.append(step)
                    timestamps.append(entry.get('timestamp', 0))

                    train_loss.append(entry.get('train/loss', float('nan')))
                    eval_loss.append(entry.get('eval/loss', float('nan')))
                    learning_rate.append(entry.get('train/lr', float('nan')))
                    grad_norm.append(entry.get('train/grad_norm', float('nan')))
                    firing_rate.append(entry.get('train/firing_rate', float('nan')))
                    sparsity.append(entry.get('train/sparsity', float('nan')))

                except json.JSONDecodeError:
                    continue

        return cls(
            name=name,
            steps=steps,
            train_loss=train_loss,
            eval_loss=eval_loss,
            learning_rate=learning_rate,
            grad_norm=grad_norm,
            firing_rate=firing_rate,
            sparsity=sparsity,
            timestamps=timestamps,
        )

    def smooth(self, values: List[float], window: int = 10) -> np.ndarray:
        """Apply exponential moving average smoothing."""
        arr = np.array(values)
        valid = ~np.isnan(arr)

        if valid.sum() < window:
            return arr

        # EMA smoothing
        alpha = 2 / (window + 1)
        smoothed = np.zeros_like(arr)
        smoothed[0] = arr[0] if valid[0] else 0

        for i in range(1, len(arr)):
            if valid[i]:
                smoothed[i] = alpha * arr[i] + (1 - alpha) * smoothed[i-1]
            else:
                smoothed[i] = smoothed[i-1]

        return smoothed


# =============================================================================
# MATPLOTLIB PLOTTING
# =============================================================================

class MatplotlibVisualizer:
    """Generate publication-quality matplotlib figures."""

    def __init__(self, style: str = 'default', figsize: Tuple[int, int] = (12, 8)):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for MatplotlibVisualizer")

        self.figsize = figsize

        # Set style
        if HAS_SEABORN:
            sns.set_theme(style='whitegrid', palette='deep')
        else:
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

        # Custom color palette
        self.colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']

    def plot_training_curves(
        self,
        runs: List[TrainingRun],
        output_path: Optional[str] = None,
        smooth_window: int = 50,
        show_raw: bool = True,
    ) -> plt.Figure:
        """
        Plot training loss curves for one or more runs.

        Creates a figure with:
        - Main loss plot (train and eval)
        - Learning rate schedule
        - Gradient norm
        """
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.25)

        # Main loss plot
        ax_loss = fig.add_subplot(gs[0, :])
        ax_lr = fig.add_subplot(gs[1, 0])
        ax_grad = fig.add_subplot(gs[1, 1])
        ax_eval = fig.add_subplot(gs[2, 0])
        ax_ppl = fig.add_subplot(gs[2, 1])

        for i, run in enumerate(runs):
            color = self.colors[i % len(self.colors)]

            # Training loss
            steps = np.array(run.steps)
            loss = np.array(run.train_loss)
            valid = ~np.isnan(loss)

            if valid.any():
                if show_raw:
                    ax_loss.plot(steps[valid], loss[valid], alpha=0.2, color=color)
                smoothed = run.smooth(run.train_loss, smooth_window)
                ax_loss.plot(steps[valid], smoothed[valid], label=f'{run.name} (train)',
                            color=color, linewidth=2)

            # Eval loss
            eval_loss = np.array(run.eval_loss)
            eval_valid = ~np.isnan(eval_loss)
            if eval_valid.any():
                ax_loss.plot(steps[eval_valid], eval_loss[eval_valid],
                            label=f'{run.name} (eval)', color=color, linestyle='--',
                            marker='o', markersize=4, linewidth=1.5)

            # Learning rate
            lr = np.array(run.learning_rate)
            lr_valid = ~np.isnan(lr)
            if lr_valid.any():
                ax_lr.plot(steps[lr_valid], lr[lr_valid], label=run.name, color=color)

            # Gradient norm
            gn = np.array(run.grad_norm)
            gn_valid = ~np.isnan(gn)
            if gn_valid.any():
                smoothed_gn = run.smooth(run.grad_norm, smooth_window)
                ax_grad.plot(steps[gn_valid], smoothed_gn[gn_valid], label=run.name, color=color)

            # Eval loss (separate plot)
            if eval_valid.any():
                ax_eval.plot(steps[eval_valid], eval_loss[eval_valid], label=run.name,
                            color=color, marker='o', markersize=4)

            # Perplexity
            if eval_valid.any():
                ppl = np.exp(np.minimum(eval_loss[eval_valid], 20))
                ax_ppl.plot(steps[eval_valid], ppl, label=run.name, color=color,
                           marker='s', markersize=4)

        # Formatting
        ax_loss.set_xlabel('Step')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax_loss.legend(loc='upper right')
        ax_loss.set_ylim(bottom=0)

        ax_lr.set_xlabel('Step')
        ax_lr.set_ylabel('Learning Rate')
        ax_lr.set_title('Learning Rate Schedule')
        ax_lr.set_yscale('log')

        ax_grad.set_xlabel('Step')
        ax_grad.set_ylabel('Gradient Norm')
        ax_grad.set_title('Gradient Norm')

        ax_eval.set_xlabel('Step')
        ax_eval.set_ylabel('Eval Loss')
        ax_eval.set_title('Evaluation Loss')

        ax_ppl.set_xlabel('Step')
        ax_ppl.set_ylabel('Perplexity')
        ax_ppl.set_title('Evaluation Perplexity')
        ax_ppl.set_yscale('log')

        plt.suptitle('BDH Training Analysis', fontsize=16, fontweight='bold', y=1.02)

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure to {output_path}")

        return fig

    def plot_biological_metrics(
        self,
        run: TrainingRun,
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot biological metrics (firing rate, sparsity)."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        steps = np.array(run.steps)

        # Firing rate
        fr = np.array(run.firing_rate)
        fr_valid = ~np.isnan(fr)
        if fr_valid.any():
            axes[0].plot(steps[fr_valid], fr[fr_valid], color=self.colors[0], alpha=0.5)
            smoothed = run.smooth(run.firing_rate, 50)
            axes[0].plot(steps[fr_valid], smoothed[fr_valid], color=self.colors[0], linewidth=2)
            axes[0].axhline(y=0.1, color='red', linestyle='--', label='Target (0.1)')
        axes[0].set_ylabel('Firing Rate')
        axes[0].set_title('Neural Firing Rate', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].set_ylim(0, 0.5)

        # Sparsity
        sp = np.array(run.sparsity)
        sp_valid = ~np.isnan(sp)
        if sp_valid.any():
            axes[1].plot(steps[sp_valid], sp[sp_valid], color=self.colors[1], alpha=0.5)
            smoothed = run.smooth(run.sparsity, 50)
            axes[1].plot(steps[sp_valid], smoothed[sp_valid], color=self.colors[1], linewidth=2)
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Sparsity')
        axes[1].set_title('Activation Sparsity', fontsize=12, fontweight='bold')
        axes[1].set_ylim(0, 1)

        plt.suptitle(f'Biological Metrics: {run.name}', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure to {output_path}")

        return fig

    def plot_weight_histograms(
        self,
        model,
        output_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot weight distribution histograms."""
        import torch

        # Collect weights by layer type
        weights_by_type = {}

        for name, param in model.named_parameters():
            if param.requires_grad and param.dim() >= 2:
                # Categorize by layer type
                if 'encoder' in name:
                    key = 'Encoder'
                elif 'decoder' in name:
                    key = 'Decoder'
                elif 'embed' in name:
                    key = 'Embedding'
                elif 'head' in name or 'lm_head' in name:
                    key = 'Output Head'
                else:
                    key = 'Other'

                if key not in weights_by_type:
                    weights_by_type[key] = []
                weights_by_type[key].append(param.detach().cpu().flatten().numpy())

        n_types = len(weights_by_type)
        fig, axes = plt.subplots(1, n_types, figsize=(4 * n_types, 4))

        if n_types == 1:
            axes = [axes]

        for ax, (name, weights_list) in zip(axes, weights_by_type.items()):
            all_weights = np.concatenate(weights_list)
            ax.hist(all_weights, bins=100, density=True, alpha=0.7, color=self.colors[0])
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Density')
            ax.set_title(f'{name} Weights')

            # Add statistics
            mean = all_weights.mean()
            std = all_weights.std()
            ax.axvline(mean, color='red', linestyle='--', label=f'μ={mean:.4f}')
            ax.legend()

        plt.suptitle('Weight Distributions', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure to {output_path}")

        return fig

    def plot_spike_raster(
        self,
        spikes: np.ndarray,
        output_path: Optional[str] = None,
        max_neurons: int = 100,
        max_timesteps: int = 500,
    ) -> plt.Figure:
        """
        Plot spike raster diagram.

        Args:
            spikes: Binary spike array [T, N] or [B, T, N]
        """
        if spikes.ndim == 3:
            spikes = spikes[0]  # Take first batch

        T, N = spikes.shape
        T = min(T, max_timesteps)
        N = min(N, max_neurons)
        spikes = spikes[:T, :N]

        fig, ax = plt.subplots(figsize=(12, 6))

        # Find spike times and neuron indices
        spike_times, neuron_ids = np.where(spikes > 0)

        ax.scatter(spike_times, neuron_ids, s=1, c='black', marker='|')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Neuron Index')
        ax.set_title('Spike Raster Plot', fontsize=14, fontweight='bold')
        ax.set_xlim(0, T)
        ax.set_ylim(0, N)

        # Add firing rate histogram on the side
        divider = ax.inset_axes([1.02, 0, 0.1, 1])
        firing_rates = spikes.mean(axis=0)
        divider.barh(range(N), firing_rates, color=self.colors[0], alpha=0.7)
        divider.set_xlabel('Rate')
        divider.set_ylim(0, N)
        divider.set_yticks([])

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure to {output_path}")

        return fig


# =============================================================================
# PLOTLY INTERACTIVE PLOTS
# =============================================================================

class PlotlyVisualizer:
    """Generate interactive Plotly figures."""

    def __init__(self):
        if not HAS_PLOTLY:
            raise ImportError("plotly is required for PlotlyVisualizer")

        self.colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']

    def plot_training_dashboard(
        self,
        runs: List[TrainingRun],
        output_path: Optional[str] = None,
    ) -> go.Figure:
        """Create interactive training dashboard."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'Learning Rate', 'Gradient Norm', 'Eval Perplexity'),
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
        )

        for i, run in enumerate(runs):
            color = self.colors[i % len(self.colors)]
            steps = run.steps

            # Training loss
            valid_loss = [v for v in run.train_loss if not math.isnan(v)]
            valid_steps = [s for s, v in zip(steps, run.train_loss) if not math.isnan(v)]
            fig.add_trace(
                go.Scatter(x=valid_steps, y=valid_loss, name=f'{run.name} (train)',
                          line=dict(color=color), legendgroup=run.name),
                row=1, col=1
            )

            # Learning rate
            valid_lr = [v for v in run.learning_rate if not math.isnan(v)]
            valid_steps_lr = [s for s, v in zip(steps, run.learning_rate) if not math.isnan(v)]
            fig.add_trace(
                go.Scatter(x=valid_steps_lr, y=valid_lr, name=f'{run.name} LR',
                          line=dict(color=color), legendgroup=run.name, showlegend=False),
                row=1, col=2
            )

            # Gradient norm
            valid_gn = [v for v in run.grad_norm if not math.isnan(v)]
            valid_steps_gn = [s for s, v in zip(steps, run.grad_norm) if not math.isnan(v)]
            fig.add_trace(
                go.Scatter(x=valid_steps_gn, y=valid_gn, name=f'{run.name} GN',
                          line=dict(color=color), legendgroup=run.name, showlegend=False),
                row=2, col=1
            )

            # Eval perplexity
            valid_eval = [(s, v) for s, v in zip(steps, run.eval_loss) if not math.isnan(v)]
            if valid_eval:
                eval_steps, eval_loss = zip(*valid_eval)
                ppl = [math.exp(min(v, 20)) for v in eval_loss]
                fig.add_trace(
                    go.Scatter(x=list(eval_steps), y=ppl, name=f'{run.name} PPL',
                              mode='lines+markers', line=dict(color=color),
                              legendgroup=run.name, showlegend=False),
                    row=2, col=2
                )

        # Update layout
        fig.update_layout(
            title='BDH Training Dashboard',
            height=700,
            showlegend=True,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        )

        fig.update_yaxes(type='log', row=1, col=2)  # Log scale for LR
        fig.update_yaxes(type='log', row=2, col=2)  # Log scale for PPL

        if output_path:
            fig.write_html(output_path)
            print(f"Saved interactive figure to {output_path}")

        return fig


# =============================================================================
# COMPARISON ANALYSIS
# =============================================================================

def compare_runs(runs: List[TrainingRun]) -> Dict:
    """Generate comparison statistics across runs."""
    comparison = {}

    for run in runs:
        # Final metrics
        valid_loss = [v for v in run.train_loss if not math.isnan(v)]
        valid_eval = [v for v in run.eval_loss if not math.isnan(v)]

        comparison[run.name] = {
            'final_train_loss': valid_loss[-1] if valid_loss else None,
            'min_train_loss': min(valid_loss) if valid_loss else None,
            'final_eval_loss': valid_eval[-1] if valid_eval else None,
            'min_eval_loss': min(valid_eval) if valid_eval else None,
            'final_perplexity': math.exp(valid_eval[-1]) if valid_eval else None,
            'total_steps': run.steps[-1] if run.steps else 0,
            'training_time': run.timestamps[-1] - run.timestamps[0] if len(run.timestamps) > 1 else 0,
        }

    return comparison


def print_comparison_table(comparison: Dict):
    """Print comparison table to console."""
    print("\n" + "=" * 80)
    print("TRAINING COMPARISON")
    print("=" * 80)

    # Header
    print(f"{'Run':<20} {'Final Loss':<12} {'Min Loss':<12} {'Perplexity':<12} {'Steps':<10}")
    print("-" * 80)

    for name, stats in comparison.items():
        final = f"{stats['final_train_loss']:.4f}" if stats['final_train_loss'] else "N/A"
        min_loss = f"{stats['min_train_loss']:.4f}" if stats['min_train_loss'] else "N/A"
        ppl = f"{stats['final_perplexity']:.2f}" if stats['final_perplexity'] else "N/A"
        steps = f"{stats['total_steps']:,}"

        print(f"{name:<20} {final:<12} {min_loss:<12} {ppl:<12} {steps:<10}")

    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="BDH Training Visualization")
    parser.add_argument("logs", nargs='+', help="Path(s) to JSONL log file(s)")
    parser.add_argument("--output", "-o", type=str, help="Output directory for figures")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg", "html"],
                       help="Output format")
    parser.add_argument("--interactive", "-i", action="store_true", help="Generate interactive Plotly plots")
    parser.add_argument("--compare", action="store_true", help="Print comparison table")
    parser.add_argument("--smooth", type=int, default=50, help="Smoothing window size")
    args = parser.parse_args()

    # Load runs
    runs = []
    for log_path in args.logs:
        print(f"Loading {log_path}...")
        run = TrainingRun.from_jsonl(log_path)
        runs.append(run)
        print(f"  Loaded {len(run.steps)} entries")

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path("./figures")
        output_dir.mkdir(exist_ok=True)

    # Generate plots
    if args.interactive and HAS_PLOTLY:
        viz = PlotlyVisualizer()
        fig = viz.plot_training_dashboard(runs, str(output_dir / "dashboard.html"))
        fig.show()
    elif HAS_MATPLOTLIB:
        viz = MatplotlibVisualizer()

        # Training curves
        output_path = output_dir / f"training_curves.{args.format}"
        viz.plot_training_curves(runs, str(output_path), smooth_window=args.smooth)

        # Biological metrics for each run
        for run in runs:
            output_path = output_dir / f"biological_{run.name}.{args.format}"
            viz.plot_biological_metrics(run, str(output_path))

        plt.show()
    else:
        print("Neither matplotlib nor plotly available for visualization")

    # Comparison
    if args.compare:
        comparison = compare_runs(runs)
        print_comparison_table(comparison)


if __name__ == "__main__":
    main()
