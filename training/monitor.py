#!/usr/bin/env python3
# Copyright 2025 - Dragon Hatching Training Monitor
# Real-time monitoring for NaN detection, gradient health, and training curves

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# ANSI COLORS
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"

    @staticmethod
    def red(s): return f"{Colors.RED}{s}{Colors.RESET}"
    @staticmethod
    def green(s): return f"{Colors.GREEN}{s}{Colors.RESET}"
    @staticmethod
    def yellow(s): return f"{Colors.YELLOW}{s}{Colors.RESET}"
    @staticmethod
    def blue(s): return f"{Colors.BLUE}{s}{Colors.RESET}"
    @staticmethod
    def bold(s): return f"{Colors.BOLD}{s}{Colors.RESET}"


# =============================================================================
# METRICS READER
# =============================================================================

class MetricsReader:
    """
    Reads training metrics from log files.

    Expects:
    - metrics.jsonl: Append-only metrics log
    - latest.json: Most recent metrics
    """

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.latest_file = self.log_dir / "latest.json"
        self.last_position = 0
        self.history = defaultdict(list)

    def read_new_metrics(self) -> List[Dict]:
        """Read new metrics since last read."""
        if not self.metrics_file.exists():
            return []

        new_records = []

        with open(self.metrics_file, "r") as f:
            f.seek(self.last_position)
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        new_records.append(record)

                        # Update history
                        for k, v in record.items():
                            if isinstance(v, (int, float)) and not math.isnan(v):
                                self.history[k].append(v)
                    except json.JSONDecodeError:
                        continue

            self.last_position = f.tell()

        return new_records

    def read_latest(self) -> Optional[Dict]:
        """Read latest metrics."""
        if not self.latest_file.exists():
            return None

        try:
            with open(self.latest_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def get_history(self, key: str, n: int = None) -> List[float]:
        """Get history for a metric."""
        values = self.history.get(key, [])
        if n:
            return values[-n:]
        return values


# =============================================================================
# ALERT SYSTEM
# =============================================================================

class AlertSystem:
    """
    Alert system for training anomalies.

    Detects:
    - NaN values
    - Gradient explosion
    - Loss spikes
    - Vanishing gradients
    - Training stalls
    """

    def __init__(self):
        self.alerts = []
        self.alert_counts = defaultdict(int)

    def check(self, metrics: Dict) -> List[Tuple[str, str, str]]:
        """
        Check metrics for anomalies.

        Returns list of (severity, alert_type, message) tuples.
        """
        alerts = []

        # NaN check
        for key in ["loss", "grad_norm_total", "firing_rate_mean"]:
            if key in metrics:
                value = metrics[key]
                if isinstance(value, float) and math.isnan(value):
                    alerts.append(("CRITICAL", "NAN", f"NaN detected in {key}"))
                    self.alert_counts["NAN"] += 1

        # Loss checks
        loss = metrics.get("loss", 0)
        if loss > 10:
            alerts.append(("CRITICAL", "LOSS_EXPLOSION", f"Loss exploded: {loss:.2f}"))
        elif loss > 5:
            alerts.append(("WARNING", "HIGH_LOSS", f"High loss: {loss:.2f}"))

        # Gradient checks
        grad_norm = metrics.get("grad_norm_total", 0)
        if grad_norm > 100:
            alerts.append(("CRITICAL", "GRAD_EXPLOSION", f"Gradient explosion: {grad_norm:.2f}"))
        elif grad_norm > 10:
            alerts.append(("WARNING", "HIGH_GRAD", f"High gradient norm: {grad_norm:.2f}"))
        elif grad_norm < 1e-7 and grad_norm > 0:
            alerts.append(("WARNING", "VANISHING_GRAD", f"Vanishing gradients: {grad_norm:.2e}"))

        # Firing rate checks
        firing_rate = metrics.get("firing_rate_mean", 0)
        if firing_rate > 0.5:
            alerts.append(("WARNING", "HIGH_FIRING", f"High firing rate: {firing_rate:.2f} (target: 0.1)"))
        elif firing_rate < 0.01 and firing_rate > 0:
            alerts.append(("WARNING", "LOW_FIRING", f"Low firing rate: {firing_rate:.3f} (target: 0.1)"))

        # Update ratio check
        update_ratio = metrics.get("update_ratio", 0)
        if update_ratio > 0.1:
            alerts.append(("WARNING", "LARGE_UPDATE", f"Large update ratio: {update_ratio:.4f}"))

        self.alerts.extend(alerts)
        return alerts

    def get_summary(self) -> Dict:
        """Get alert summary."""
        return {
            "total_alerts": len(self.alerts),
            "by_type": dict(self.alert_counts),
            "recent": self.alerts[-10:] if self.alerts else [],
        }


# =============================================================================
# TRAINING CURVES
# =============================================================================

class TrainingCurves:
    """
    Manages training curve visualization.

    Can use:
    - Matplotlib (if available)
    - ASCII art (fallback)
    """

    def __init__(self, use_matplotlib: bool = True):
        self.use_matplotlib = use_matplotlib and HAS_MATPLOTLIB
        self.fig = None
        self.axes = None

    def setup_matplotlib(self):
        """Setup matplotlib figure."""
        if not self.use_matplotlib:
            return

        plt.ion()
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle("Dragon Hatching Training Monitor", fontsize=14)

        # Configure subplots
        titles = ["Loss", "Gradient Norm", "Firing Rate", "Learning Rate"]
        for ax, title in zip(self.axes.flat, titles):
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

    def update_matplotlib(self, history: Dict):
        """Update matplotlib plots."""
        if not self.use_matplotlib or self.fig is None:
            return

        # Loss
        ax = self.axes[0, 0]
        ax.clear()
        ax.set_title("Loss")
        if "loss" in history:
            ax.plot(history["loss"], "b-", label="Train")
        if "eval_val_loss" in history:
            ax.plot(history["eval_val_loss"], "r-", label="Val")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Gradient norm
        ax = self.axes[0, 1]
        ax.clear()
        ax.set_title("Gradient Norm")
        if "grad_norm_total" in history:
            ax.plot(history["grad_norm_total"], "g-")
            ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="Clip threshold")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        # Firing rate
        ax = self.axes[1, 0]
        ax.clear()
        ax.set_title("Firing Rate")
        if "firing_rate_mean" in history:
            ax.plot(history["firing_rate_mean"], "m-")
            ax.axhline(y=0.1, color="r", linestyle="--", alpha=0.5, label="Target")
        ax.set_ylim(0, 0.5)
        ax.grid(True, alpha=0.3)

        # Learning rate
        ax = self.axes[1, 1]
        ax.clear()
        ax.set_title("Learning Rate")
        if "lr" in history:
            ax.plot(history["lr"], "c-")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def ascii_plot(self, values: List[float], width: int = 50, height: int = 10, title: str = "") -> str:
        """Create ASCII art plot."""
        if not values:
            return f"{title}: No data"

        # Normalize values
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val > min_val else 1

        # Sample values if too many
        if len(values) > width:
            step = len(values) // width
            values = values[::step][:width]

        # Create plot
        lines = []
        lines.append(f"┌{'─' * (width + 2)}┐ {title}")
        lines.append(f"│ {max_val:>8.4f} │")

        for row in range(height):
            threshold = max_val - (row + 0.5) * range_val / height
            line = "│ "
            for v in values:
                if v >= threshold:
                    line += "█"
                else:
                    line += " "
            line += " │"
            lines.append(line)

        lines.append(f"│ {min_val:>8.4f} │")
        lines.append(f"└{'─' * (width + 2)}┘")

        return "\n".join(lines)

    def show_ascii(self, history: Dict):
        """Show ASCII plots."""
        plots = []

        if "loss" in history and history["loss"]:
            plots.append(self.ascii_plot(history["loss"][-100:], title="Loss"))

        if "grad_norm_total" in history and history["grad_norm_total"]:
            plots.append(self.ascii_plot(history["grad_norm_total"][-100:], title="Grad Norm"))

        if "firing_rate_mean" in history and history["firing_rate_mean"]:
            plots.append(self.ascii_plot(history["firing_rate_mean"][-100:], title="Firing Rate"))

        return "\n\n".join(plots)


# =============================================================================
# MONITOR
# =============================================================================

class TrainingMonitor:
    """
    Complete training monitor.

    Features:
    - Real-time metrics display
    - NaN detection with alerts
    - Training curves (matplotlib or ASCII)
    - Alert system
    - Export to CSV
    """

    def __init__(
        self,
        log_dir: str,
        refresh_interval: float = 5.0,
        use_matplotlib: bool = True,
        export_csv: str = None,
    ):
        self.log_dir = Path(log_dir)
        self.refresh_interval = refresh_interval
        self.export_csv = export_csv

        self.reader = MetricsReader(log_dir)
        self.alerts = AlertSystem()
        self.curves = TrainingCurves(use_matplotlib)

        self.running = False

    def format_metric(self, key: str, value: float) -> str:
        """Format a metric for display."""
        if math.isnan(value):
            return Colors.red(f"{key}: NaN")

        # Color based on health
        if key == "loss":
            if value > 5:
                return Colors.red(f"{key}: {value:.4f}")
            elif value < 2:
                return Colors.green(f"{key}: {value:.4f}")
            return f"{key}: {value:.4f}"

        if key == "grad_norm_total":
            if value > 10:
                return Colors.red(f"{key}: {value:.2f}")
            elif value < 0.001:
                return Colors.yellow(f"{key}: {value:.2e}")
            return Colors.green(f"{key}: {value:.2f}")

        if key == "firing_rate_mean":
            if abs(value - 0.1) < 0.05:
                return Colors.green(f"{key}: {value:.3f}")
            return Colors.yellow(f"{key}: {value:.3f}")

        if key == "lr":
            return f"{key}: {value:.2e}"

        if isinstance(value, float):
            if abs(value) < 0.001:
                return f"{key}: {value:.2e}"
            return f"{key}: {value:.4f}"

        return f"{key}: {value}"

    def display_status(self, latest: Dict):
        """Display current training status."""
        # Clear screen
        print("\033[2J\033[H", end="")

        # Header
        print(Colors.bold("=" * 60))
        print(Colors.bold(f"  🐉 DRAGON HATCHING TRAINING MONITOR"))
        print(Colors.bold("=" * 60))
        print()

        if not latest:
            print(Colors.yellow("Waiting for training data..."))
            print(f"Monitoring: {self.log_dir}")
            return

        # Step info
        step = latest.get("step", 0)
        elapsed = latest.get("elapsed", 0)
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

        print(f"Step: {Colors.bold(str(step))} | Elapsed: {elapsed_str}")
        print(f"Stage: {latest.get('stage', '?')}")
        print()

        # Key metrics
        print(Colors.bold("📊 Key Metrics:"))
        print("-" * 40)

        key_metrics = ["loss", "val_loss", "lr", "grad_norm_total", "firing_rate_mean"]
        for key in key_metrics:
            if key in latest:
                print(f"  {self.format_metric(key, latest[key])}")

        print()

        # Gradient health
        print(Colors.bold("🔬 Gradient Health:"))
        print("-" * 40)
        grad_metrics = ["grad_norm_mean", "grad_norm_max", "grad_max", "update_ratio"]
        for key in grad_metrics:
            if key in latest:
                print(f"  {self.format_metric(key, latest[key])}")

        print()

        # Biological metrics
        print(Colors.bold("🧠 Biological Metrics:"))
        print("-" * 40)
        bio_metrics = [k for k in latest.keys() if "firing" in k or "active" in k]
        for key in bio_metrics[:5]:  # Limit display
            print(f"  {self.format_metric(key, latest[key])}")

        print()

        # Alerts
        recent_alerts = self.alerts.check(latest)
        if recent_alerts:
            print(Colors.bold("⚠️  Alerts:"))
            print("-" * 40)
            for severity, alert_type, message in recent_alerts[-5:]:
                if severity == "CRITICAL":
                    print(f"  {Colors.red('🔴 ' + message)}")
                else:
                    print(f"  {Colors.yellow('🟡 ' + message)}")
        else:
            print(Colors.green("✅ No alerts - training healthy"))

        print()
        print("-" * 60)
        print(f"Refresh: {self.refresh_interval}s | Press Ctrl+C to stop")

    def export_to_csv(self):
        """Export metrics history to CSV."""
        if not self.export_csv:
            return

        history = self.reader.history

        # Get all keys
        all_keys = sorted(set(k for k in history.keys()))

        # Write CSV
        with open(self.export_csv, "w") as f:
            # Header
            f.write(",".join(all_keys) + "\n")

            # Data
            max_len = max(len(v) for v in history.values()) if history else 0
            for i in range(max_len):
                row = []
                for key in all_keys:
                    values = history.get(key, [])
                    if i < len(values):
                        row.append(str(values[i]))
                    else:
                        row.append("")
                f.write(",".join(row) + "\n")

        print(f"Exported metrics to {self.export_csv}")

    def run(self):
        """Run the monitor."""
        self.running = True

        # Setup matplotlib if available
        if self.curves.use_matplotlib:
            self.curves.setup_matplotlib()

        try:
            while self.running:
                # Read new metrics
                self.reader.read_new_metrics()
                latest = self.reader.read_latest()

                # Display status
                self.display_status(latest)

                # Update curves
                if self.curves.use_matplotlib:
                    self.curves.update_matplotlib(self.reader.history)
                else:
                    # Show ASCII plots
                    print()
                    print(self.curves.show_ascii(self.reader.history))

                # Sleep
                time.sleep(self.refresh_interval)

        except KeyboardInterrupt:
            print("\nMonitor stopped.")

        finally:
            self.running = False

            # Export if requested
            if self.export_csv:
                self.export_to_csv()

            # Show summary
            print()
            print(Colors.bold("📈 Session Summary:"))
            summary = self.alerts.get_summary()
            print(f"  Total alerts: {summary['total_alerts']}")
            for alert_type, count in summary["by_type"].items():
                print(f"    {alert_type}: {count}")


# =============================================================================
# CLI
# =============================================================================

def find_latest_log_dir(base_dir: str = "logs") -> Optional[Path]:
    """Find the most recent log directory."""
    base = Path(base_dir)
    if not base.exists():
        return None

    dirs = [d for d in base.iterdir() if d.is_dir()]
    if not dirs:
        return None

    # Sort by modification time
    dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return dirs[0]


def main():
    parser = argparse.ArgumentParser(description="Dragon Hatching Training Monitor")

    parser.add_argument(
        "--log-dir", type=str, default=None,
        help="Log directory to monitor (default: latest in logs/)"
    )
    parser.add_argument(
        "--refresh", type=float, default=5.0,
        help="Refresh interval in seconds (default: 5)"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable matplotlib plotting"
    )
    parser.add_argument(
        "--export", type=str, default=None,
        help="Export metrics to CSV file"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run once and exit (don't loop)"
    )

    args = parser.parse_args()

    # Find log directory
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = find_latest_log_dir()

    if not log_dir or not log_dir.exists():
        print(Colors.red("No log directory found!"))
        print("Run training first or specify --log-dir")
        sys.exit(1)

    print(f"Monitoring: {log_dir}")

    # Create monitor
    monitor = TrainingMonitor(
        log_dir=str(log_dir),
        refresh_interval=args.refresh,
        use_matplotlib=not args.no_plot,
        export_csv=args.export,
    )

    if args.once:
        # Run once
        monitor.reader.read_new_metrics()
        latest = monitor.reader.read_latest()
        monitor.display_status(latest)

        if not args.no_plot and HAS_MATPLOTLIB:
            monitor.curves.setup_matplotlib()
            monitor.curves.update_matplotlib(monitor.reader.history)
            plt.show()
    else:
        # Run continuously
        monitor.run()


if __name__ == "__main__":
    main()
