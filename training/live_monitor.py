#!/usr/bin/env python3
# Copyright 2025 - Live Training Monitor for Dragon Hatching
#
# Real-time monitoring dashboard with:
# - Live loss curves
# - Learning rate schedule visualization
# - Gradient statistics
# - Biological metrics (firing rate, sparsity)
# - GPU/CPU utilization
# - ETA estimation
# - Terminal-based UI (no browser needed)

import argparse
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import math


# =============================================================================
# TERMINAL UI UTILITIES
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_BLUE = '\033[44m'

    @staticmethod
    def rgb(r: int, g: int, b: int) -> str:
        return f'\033[38;2;{r};{g};{b}m'


def clear_screen():
    """Clear terminal screen."""
    print('\033[2J\033[H', end='')


def move_cursor(row: int, col: int):
    """Move cursor to position."""
    print(f'\033[{row};{col}H', end='')


def hide_cursor():
    """Hide terminal cursor."""
    print('\033[?25l', end='')


def show_cursor():
    """Show terminal cursor."""
    print('\033[?25h', end='')


# =============================================================================
# SPARKLINE CHARTS
# =============================================================================

def sparkline(values: List[float], width: int = 50, min_val: Optional[float] = None,
              max_val: Optional[float] = None) -> str:
    """
    Create a sparkline chart from values.

    Uses Unicode block characters for sub-character resolution.
    """
    if not values:
        return ' ' * width

    # Normalize to width
    if len(values) > width:
        # Downsample
        step = len(values) / width
        values = [values[int(i * step)] for i in range(width)]
    elif len(values) < width:
        # Pad with last value
        values = values + [values[-1]] * (width - len(values))

    # Get range
    if min_val is None:
        min_val = min(values)
    if max_val is None:
        max_val = max(values)

    if max_val == min_val:
        max_val = min_val + 1

    # Unicode block characters (8 levels)
    blocks = ' ▁▂▃▄▅▆▇█'

    result = []
    for v in values:
        # Normalize to 0-8
        normalized = (v - min_val) / (max_val - min_val)
        idx = min(8, max(0, int(normalized * 8)))
        result.append(blocks[idx])

    return ''.join(result)


def bar_chart(value: float, max_value: float, width: int = 30,
              color: str = Colors.GREEN) -> str:
    """Create a horizontal bar chart."""
    if max_value <= 0:
        return ' ' * width

    filled = int((value / max_value) * width)
    filled = min(width, max(0, filled))

    bar = '█' * filled + '░' * (width - filled)
    return f"{color}{bar}{Colors.RESET}"


# =============================================================================
# LOG FILE PARSER
# =============================================================================

@dataclass
class TrainingState:
    """Current training state parsed from logs."""
    step: int = 0
    max_steps: int = 100000
    loss: float = 0.0
    eval_loss: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    firing_rate: float = 0.0
    sparsity: float = 0.0
    timestamp: float = 0.0

    # History for charts
    loss_history: List[float] = None
    eval_loss_history: List[float] = None
    lr_history: List[float] = None
    grad_norm_history: List[float] = None

    def __post_init__(self):
        if self.loss_history is None:
            self.loss_history = []
        if self.eval_loss_history is None:
            self.eval_loss_history = []
        if self.lr_history is None:
            self.lr_history = []
        if self.grad_norm_history is None:
            self.grad_norm_history = []


class LogParser:
    """Parse training logs from JSONL file."""

    def __init__(self, log_path: str, max_history: int = 1000):
        self.log_path = Path(log_path)
        self.max_history = max_history
        self.last_position = 0
        self.state = TrainingState()

    def update(self) -> TrainingState:
        """Parse new log entries and update state."""
        if not self.log_path.exists():
            return self.state

        with open(self.log_path, 'r') as f:
            f.seek(self.last_position)
            new_lines = f.readlines()
            self.last_position = f.tell()

        for line in new_lines:
            try:
                entry = json.loads(line.strip())
                self._process_entry(entry)
            except json.JSONDecodeError:
                continue

        return self.state

    def _process_entry(self, entry: Dict):
        """Process a single log entry."""
        self.state.step = entry.get('step', self.state.step)
        self.state.timestamp = entry.get('timestamp', time.time())

        if 'train/loss' in entry:
            self.state.loss = entry['train/loss']
            self.state.loss_history.append(self.state.loss)
            if len(self.state.loss_history) > self.max_history:
                self.state.loss_history.pop(0)

        if 'eval/loss' in entry:
            self.state.eval_loss = entry['eval/loss']
            self.state.eval_loss_history.append(self.state.eval_loss)
            if len(self.state.eval_loss_history) > self.max_history:
                self.state.eval_loss_history.pop(0)

        if 'train/lr' in entry:
            self.state.learning_rate = entry['train/lr']
            self.state.lr_history.append(self.state.learning_rate)
            if len(self.state.lr_history) > self.max_history:
                self.state.lr_history.pop(0)

        if 'train/grad_norm' in entry:
            self.state.grad_norm = entry['train/grad_norm']
            self.state.grad_norm_history.append(self.state.grad_norm)
            if len(self.state.grad_norm_history) > self.max_history:
                self.state.grad_norm_history.pop(0)

        if 'train/firing_rate' in entry:
            self.state.firing_rate = entry['train/firing_rate']

        if 'train/sparsity' in entry:
            self.state.sparsity = entry['train/sparsity']


# =============================================================================
# DASHBOARD
# =============================================================================

class Dashboard:
    """Terminal-based training dashboard."""

    def __init__(self, log_path: str, max_steps: int = 100000):
        self.parser = LogParser(log_path)
        self.parser.state.max_steps = max_steps
        self.start_time = time.time()
        self.start_step = 0

    def format_time(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        if seconds < 0 or math.isinf(seconds):
            return "--:--:--"
        return str(timedelta(seconds=int(seconds)))

    def estimate_eta(self, state: TrainingState) -> str:
        """Estimate time to completion."""
        if state.step <= self.start_step:
            return "--:--:--"

        elapsed = time.time() - self.start_time
        steps_done = state.step - self.start_step
        steps_remaining = state.max_steps - state.step

        if steps_done <= 0:
            return "--:--:--"

        time_per_step = elapsed / steps_done
        eta_seconds = time_per_step * steps_remaining

        return self.format_time(eta_seconds)

    def render_header(self, state: TrainingState) -> str:
        """Render dashboard header."""
        lines = []

        # Title
        title = f"  🐉 BDH Training Monitor"
        lines.append(f"{Colors.BOLD}{Colors.CYAN}{title}{Colors.RESET}")
        lines.append(f"{Colors.DIM}{'─' * 70}{Colors.RESET}")

        return '\n'.join(lines)

    def render_progress(self, state: TrainingState) -> str:
        """Render progress section."""
        lines = []

        # Progress bar
        progress = state.step / max(state.max_steps, 1)
        progress_bar = bar_chart(state.step, state.max_steps, width=50)
        pct = progress * 100

        lines.append(f"  {Colors.BOLD}Progress:{Colors.RESET}")
        lines.append(f"  {progress_bar} {pct:5.1f}%")
        lines.append(f"  Step: {state.step:,} / {state.max_steps:,}")

        # Time estimates
        elapsed = time.time() - self.start_time
        eta = self.estimate_eta(state)
        lines.append(f"  Elapsed: {self.format_time(elapsed)}  |  ETA: {eta}")

        return '\n'.join(lines)

    def render_metrics(self, state: TrainingState) -> str:
        """Render metrics section."""
        lines = []

        lines.append(f"\n  {Colors.BOLD}Metrics:{Colors.RESET}")

        # Loss
        loss_color = Colors.GREEN if state.loss < 5 else Colors.YELLOW if state.loss < 10 else Colors.RED
        lines.append(f"  Train Loss:  {loss_color}{state.loss:8.4f}{Colors.RESET}")

        if state.eval_loss > 0:
            lines.append(f"  Eval Loss:   {state.eval_loss:8.4f}")

        # Perplexity
        if state.loss > 0:
            ppl = math.exp(min(state.loss, 20))
            lines.append(f"  Perplexity:  {ppl:8.2f}")

        # Learning rate
        lines.append(f"  LR:          {state.learning_rate:8.2e}")

        # Gradient norm
        if state.grad_norm > 0:
            gn_color = Colors.GREEN if state.grad_norm < 1 else Colors.YELLOW if state.grad_norm < 10 else Colors.RED
            lines.append(f"  Grad Norm:   {gn_color}{state.grad_norm:8.4f}{Colors.RESET}")

        # Biological metrics
        if state.firing_rate > 0:
            lines.append(f"  Firing Rate: {state.firing_rate:8.4f}")
        if state.sparsity > 0:
            lines.append(f"  Sparsity:    {state.sparsity:8.4f}")

        return '\n'.join(lines)

    def render_charts(self, state: TrainingState) -> str:
        """Render sparkline charts."""
        lines = []

        lines.append(f"\n  {Colors.BOLD}Charts:{Colors.RESET}")

        # Loss history
        if state.loss_history:
            chart = sparkline(state.loss_history, width=50)
            min_loss = min(state.loss_history)
            max_loss = max(state.loss_history)
            lines.append(f"  Loss:     {Colors.CYAN}{chart}{Colors.RESET} [{min_loss:.2f}-{max_loss:.2f}]")

        # Eval loss history
        if state.eval_loss_history:
            chart = sparkline(state.eval_loss_history, width=50)
            min_loss = min(state.eval_loss_history)
            max_loss = max(state.eval_loss_history)
            lines.append(f"  Eval:     {Colors.MAGENTA}{chart}{Colors.RESET} [{min_loss:.2f}-{max_loss:.2f}]")

        # LR history
        if state.lr_history:
            chart = sparkline(state.lr_history, width=50)
            lines.append(f"  LR:       {Colors.YELLOW}{chart}{Colors.RESET}")

        # Grad norm history
        if state.grad_norm_history:
            chart = sparkline(state.grad_norm_history, width=50)
            lines.append(f"  GradNorm: {Colors.GREEN}{chart}{Colors.RESET}")

        return '\n'.join(lines)

    def render_footer(self) -> str:
        """Render dashboard footer."""
        lines = []
        lines.append(f"\n{Colors.DIM}{'─' * 70}{Colors.RESET}")
        lines.append(f"  {Colors.DIM}Press Ctrl+C to exit | Updated: {datetime.now().strftime('%H:%M:%S')}{Colors.RESET}")
        return '\n'.join(lines)

    def render(self) -> str:
        """Render full dashboard."""
        state = self.parser.update()

        if self.start_step == 0 and state.step > 0:
            self.start_step = state.step
            self.start_time = time.time()

        sections = [
            self.render_header(state),
            self.render_progress(state),
            self.render_metrics(state),
            self.render_charts(state),
            self.render_footer(),
        ]

        return '\n'.join(sections)

    def run(self, refresh_rate: float = 1.0):
        """Run the dashboard loop."""
        hide_cursor()

        try:
            while True:
                clear_screen()
                print(self.render())
                time.sleep(refresh_rate)
        except KeyboardInterrupt:
            pass
        finally:
            show_cursor()
            print("\nMonitor stopped.")


# =============================================================================
# SYSTEM MONITOR
# =============================================================================

def get_gpu_info() -> Optional[Dict]:
    """Get GPU utilization info."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                'name': torch.cuda.get_device_name(0),
                'memory_used': torch.cuda.memory_allocated(0) / 1e9,
                'memory_total': torch.cuda.get_device_properties(0).total_memory / 1e9,
                'utilization': torch.cuda.utilization(0) if hasattr(torch.cuda, 'utilization') else None,
            }
    except Exception:
        pass
    return None


def get_cpu_info() -> Dict:
    """Get CPU utilization info."""
    try:
        import psutil
        return {
            'percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
        }
    except ImportError:
        return {}


# =============================================================================
# MAIN
# =============================================================================

def find_latest_log(log_dir: str) -> Optional[str]:
    """Find the most recent log file in directory."""
    log_path = Path(log_dir)
    if not log_path.exists():
        return None

    jsonl_files = list(log_path.glob("*.jsonl"))
    if not jsonl_files:
        return None

    return str(max(jsonl_files, key=lambda p: p.stat().st_mtime))


def main():
    parser = argparse.ArgumentParser(description="BDH Training Monitor")
    parser.add_argument("--log", type=str, help="Path to JSONL log file")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Log directory")
    parser.add_argument("--max-steps", type=int, default=100000, help="Maximum training steps")
    parser.add_argument("--refresh", type=float, default=1.0, help="Refresh rate in seconds")
    args = parser.parse_args()

    # Find log file
    log_path = args.log
    if not log_path:
        log_path = find_latest_log(args.log_dir)
        if not log_path:
            print(f"No log files found in {args.log_dir}")
            print("Start training first, or specify --log path")
            sys.exit(1)

    print(f"Monitoring: {log_path}")
    time.sleep(1)

    # Run dashboard
    dashboard = Dashboard(log_path, args.max_steps)
    dashboard.run(args.refresh)


if __name__ == "__main__":
    main()
