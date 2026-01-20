# Copyright 2025 - Dataset Loaders and Recommendations for Dragon Hatching
# Curated datasets for text coherence, function calling, tool use, and bash

"""
DATASET RECOMMENDATIONS FOR DRAGON HATCHING
============================================

This module provides loaders and utilities for recommended datasets.
Each dataset is chosen for specific training objectives.

TRAINING STAGES & RECOMMENDED DATASETS:

Stage 1: Text Coherence & Language Understanding
------------------------------------------------
- TinyStories (10M tokens) - Simple coherent narratives
- OpenWebText (subset) - Web text diversity
- WikiText-103 - Factual, structured text

Stage 2: Function Calling
-------------------------
- Berkeley Function Calling Leaderboard (BFCL) - Gold standard
- Gorilla API Bench - Real API usage patterns
- ToolACE - Diverse tool scenarios
- Seal-Tools - Nested function calls

Stage 3: Bash Commands
----------------------
- NL2Bash (~10K examples) - Bash one-liners with descriptions
- unix-commands (HuggingFace) - Unix command patterns
- bash-commands-dataset - NL to bash pairs

Stage 4: Code & Instruction Following
-------------------------------------
- OpenCodeInstruct (5M samples) - Largest open code dataset
- CodeAlpaca - Code instruction tuning
- CoMIT - Multilingual code instructions

Stage 5: Tool Use & Agents
--------------------------
- ToolBench - Multi-step tool usage
- AgentInstruct - Agent behavior patterns
- WebArena (for evaluation) - Real web tasks
"""

import os
import json
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Iterator, Tuple
from pathlib import Path
import urllib.request

# Try to import huggingface datasets
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: 'datasets' library not installed. Some loaders unavailable.")


# =============================================================================
# DATASET REGISTRY
# =============================================================================

@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    description: str
    url: str
    size: str
    task_types: List[str]
    license: str
    hf_path: Optional[str] = None  # HuggingFace path if available
    download_url: Optional[str] = None  # Direct download if available
    priority: int = 1  # 1=essential, 2=recommended, 3=optional


DATASET_REGISTRY = {
    # =========================================================================
    # TEXT COHERENCE
    # =========================================================================
    "tinystories": DatasetInfo(
        name="TinyStories",
        description="Synthetic stories for coherent narrative generation",
        url="https://huggingface.co/datasets/roneneldan/TinyStories",
        size="~500MB",
        task_types=["text_completion", "coherence"],
        license="MIT",
        hf_path="roneneldan/TinyStories",
        priority=1
    ),
    "wikitext": DatasetInfo(
        name="WikiText-103",
        description="Wikipedia articles for factual text understanding",
        url="https://huggingface.co/datasets/wikitext",
        size="~500MB",
        task_types=["text_completion", "coherence"],
        license="CC-BY-SA-3.0",
        hf_path="wikitext",
        priority=2
    ),
    "openwebtext": DatasetInfo(
        name="OpenWebText",
        description="Web text corpus (GPT-2 training data recreation)",
        url="https://huggingface.co/datasets/openwebtext",
        size="~40GB",
        task_types=["text_completion", "coherence"],
        license="MIT",
        hf_path="openwebtext",
        priority=3
    ),

    # =========================================================================
    # FUNCTION CALLING
    # =========================================================================
    "bfcl": DatasetInfo(
        name="Berkeley Function Calling Leaderboard",
        description="Gold standard for function calling evaluation",
        url="https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        size="~50MB",
        task_types=["function_call"],
        license="Apache-2.0",
        hf_path="gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        priority=1
    ),
    "gorilla_apibench": DatasetInfo(
        name="Gorilla APIBench",
        description="API call dataset from Gorilla project",
        url="https://github.com/ShishirPatil/gorilla",
        size="~100MB",
        task_types=["function_call", "tool_use"],
        license="Apache-2.0",
        hf_path="gorilla-llm/APIBench",
        priority=1
    ),
    "toolace": DatasetInfo(
        name="ToolACE",
        description="Diverse tool-learning scenarios",
        url="https://arxiv.org/abs/2409.00920",
        size="~200MB",
        task_types=["function_call", "tool_use"],
        license="Apache-2.0",
        hf_path=None,  # Check for availability
        priority=2
    ),
    "glaive_function_calling": DatasetInfo(
        name="Glaive Function Calling v2",
        description="113K function calling conversations",
        url="https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2",
        size="~150MB",
        task_types=["function_call"],
        license="Apache-2.0",
        hf_path="glaiveai/glaive-function-calling-v2",
        priority=1
    ),

    # =========================================================================
    # BASH COMMANDS
    # =========================================================================
    "nl2bash": DatasetInfo(
        name="NL2Bash",
        description="~10K bash one-liners with English descriptions",
        url="https://github.com/TellinaTool/nl2bash",
        size="~5MB",
        task_types=["bash_command"],
        license="MIT",
        hf_path=None,
        download_url="https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash/all.cm",
        priority=1
    ),
    "unix_commands": DatasetInfo(
        name="Unix Commands Dataset",
        description="Unix command line usage examples",
        url="https://huggingface.co/datasets/harpomaxx/unix-commands",
        size="~10MB",
        task_types=["bash_command"],
        license="MIT",
        hf_path="harpomaxx/unix-commands",
        priority=2
    ),
    "bash_commands": DatasetInfo(
        name="Bash Commands Dataset",
        description="NL prompts paired with bash commands",
        url="https://huggingface.co/datasets/aelhalili/bash-commands-dataset",
        size="~5MB",
        task_types=["bash_command"],
        license="MIT",
        hf_path="aelhalili/bash-commands-dataset",
        priority=2
    ),

    # =========================================================================
    # CODE INSTRUCTION
    # =========================================================================
    "opencodeinstruct": DatasetInfo(
        name="OpenCodeInstruct",
        description="5M code instruction samples with tests",
        url="https://arxiv.org/abs/2504.04030",
        size="~10GB",
        task_types=["code", "chain_of_thought"],
        license="Apache-2.0",
        hf_path="OpenCoder-LLM/opc-sft-stage1",
        priority=2
    ),
    "code_alpaca": DatasetInfo(
        name="Code Alpaca",
        description="20K code instruction-following examples",
        url="https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k",
        size="~20MB",
        task_types=["code", "chain_of_thought"],
        license="Apache-2.0",
        hf_path="sahil2801/CodeAlpaca-20k",
        priority=1
    ),

    # =========================================================================
    # TOOL USE / AGENTS
    # =========================================================================
    "toolbench": DatasetInfo(
        name="ToolBench",
        description="Multi-step tool usage training data",
        url="https://github.com/OpenBMB/ToolBench",
        size="~500MB",
        task_types=["tool_use", "function_call"],
        license="Apache-2.0",
        hf_path="ToolBench/ToolBench",
        priority=1
    ),
    "agentinstruct": DatasetInfo(
        name="AgentInstruct",
        description="25K agent interaction examples",
        url="https://huggingface.co/datasets/THUDM/AgentInstruct",
        size="~100MB",
        task_types=["tool_use", "chain_of_thought"],
        license="Apache-2.0",
        hf_path="THUDM/AgentInstruct",
        priority=2
    ),

    # =========================================================================
    # CHAIN OF THOUGHT
    # =========================================================================
    "gsm8k": DatasetInfo(
        name="GSM8K",
        description="Grade school math with step-by-step solutions",
        url="https://huggingface.co/datasets/gsm8k",
        size="~10MB",
        task_types=["chain_of_thought"],
        license="MIT",
        hf_path="gsm8k",
        priority=1
    ),
    "aqua_rat": DatasetInfo(
        name="AQuA-RAT",
        description="Math word problems with rationales",
        url="https://huggingface.co/datasets/aqua_rat",
        size="~20MB",
        task_types=["chain_of_thought"],
        license="Apache-2.0",
        hf_path="aqua_rat",
        priority=2
    ),
}


# =============================================================================
# DATASET LOADERS
# =============================================================================

class BaseDatasetLoader:
    """Base class for dataset loaders."""

    def __init__(self, cache_dir: str = "./data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> Iterator[Dict]:
        """Load dataset and yield samples."""
        raise NotImplementedError

    def to_training_format(self, sample: Dict) -> Dict:
        """Convert sample to training format."""
        raise NotImplementedError


class TinyStoriesLoader(BaseDatasetLoader):
    """Loader for TinyStories dataset."""

    def load(self, split: str = "train", max_samples: int = None) -> Iterator[Dict]:
        if not HF_AVAILABLE:
            raise ImportError("Install 'datasets' library: pip install datasets")

        dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)

        for i, sample in enumerate(dataset):
            if max_samples and i >= max_samples:
                break

            yield self.to_training_format(sample)

    def to_training_format(self, sample: Dict) -> Dict:
        text = sample.get("text", "")
        # Split into prompt/completion
        sentences = text.split(". ")
        if len(sentences) > 2:
            split_point = len(sentences) // 2
            prompt = ". ".join(sentences[:split_point]) + "."
            completion = ". ".join(sentences[split_point:])
        else:
            prompt = text[:len(text)//2]
            completion = text[len(text)//2:]

        return {
            "input": f"Continue the story:\n{prompt}\n",
            "output": completion,
            "task_type": "text_completion"
        }


class BFCLLoader(BaseDatasetLoader):
    """Loader for Berkeley Function Calling Leaderboard."""

    def load(self, split: str = "train", max_samples: int = None) -> Iterator[Dict]:
        if not HF_AVAILABLE:
            raise ImportError("Install 'datasets' library: pip install datasets")

        dataset = load_dataset(
            "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
            split=split,
            streaming=True
        )

        for i, sample in enumerate(dataset):
            if max_samples and i >= max_samples:
                break

            formatted = self.to_training_format(sample)
            if formatted:
                yield formatted

    def to_training_format(self, sample: Dict) -> Optional[Dict]:
        try:
            question = sample.get("question", "")
            # Parse function info if available
            functions = sample.get("function", [])

            if not question:
                return None

            # Format as function calling
            func_str = json.dumps(functions) if functions else "[]"

            return {
                "input": f"Available functions: {func_str}\n\nUser: {question}\nAssistant:",
                "output": sample.get("ground_truth", ""),
                "task_type": "function_call"
            }
        except Exception:
            return None


class NL2BashLoader(BaseDatasetLoader):
    """Loader for NL2Bash dataset."""

    def load(self, max_samples: int = None) -> Iterator[Dict]:
        # Download if not cached
        cache_file = self.cache_dir / "nl2bash.json"

        if not cache_file.exists():
            self._download()

        with open(cache_file) as f:
            data = json.load(f)

        for i, sample in enumerate(data):
            if max_samples and i >= max_samples:
                break
            yield self.to_training_format(sample)

    def _download(self):
        """Download and parse NL2Bash data."""
        urls = [
            "https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash/all.nl",
            "https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash/all.cm",
        ]

        nl_file = self.cache_dir / "all.nl"
        cm_file = self.cache_dir / "all.cm"

        # Download files
        for url, filepath in [(urls[0], nl_file), (urls[1], cm_file)]:
            if not filepath.exists():
                print(f"Downloading {url}...")
                urllib.request.urlretrieve(url, filepath)

        # Parse and combine
        with open(nl_file) as f:
            descriptions = f.readlines()
        with open(cm_file) as f:
            commands = f.readlines()

        data = []
        for desc, cmd in zip(descriptions, commands):
            desc = desc.strip()
            cmd = cmd.strip()
            if desc and cmd:
                data.append({"description": desc, "command": cmd})

        # Save as JSON
        cache_file = self.cache_dir / "nl2bash.json"
        with open(cache_file, "w") as f:
            json.dump(data, f)

        print(f"Saved {len(data)} NL2Bash examples to {cache_file}")

    def to_training_format(self, sample: Dict) -> Dict:
        return {
            "input": f"User: {sample['description']}\nAssistant:",
            "output": f"```bash\n{sample['command']}\n```",
            "task_type": "bash_command"
        }


class GlaiveFunctionCallingLoader(BaseDatasetLoader):
    """Loader for Glaive Function Calling dataset."""

    def load(self, split: str = "train", max_samples: int = None) -> Iterator[Dict]:
        if not HF_AVAILABLE:
            raise ImportError("Install 'datasets' library: pip install datasets")

        dataset = load_dataset(
            "glaiveai/glaive-function-calling-v2",
            split=split,
            streaming=True
        )

        for i, sample in enumerate(dataset):
            if max_samples and i >= max_samples:
                break

            formatted = self.to_training_format(sample)
            if formatted:
                yield formatted

    def to_training_format(self, sample: Dict) -> Optional[Dict]:
        try:
            system = sample.get("system", "")
            chat = sample.get("chat", "")

            if not chat:
                return None

            return {
                "input": f"System: {system}\n\n{chat.split('GPT4')[0]}",
                "output": chat.split("GPT4")[-1] if "GPT4" in chat else chat,
                "task_type": "function_call"
            }
        except Exception:
            return None


class GSM8KLoader(BaseDatasetLoader):
    """Loader for GSM8K (chain of thought)."""

    def load(self, split: str = "train", max_samples: int = None) -> Iterator[Dict]:
        if not HF_AVAILABLE:
            raise ImportError("Install 'datasets' library: pip install datasets")

        dataset = load_dataset("gsm8k", "main", split=split, streaming=True)

        for i, sample in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            yield self.to_training_format(sample)

    def to_training_format(self, sample: Dict) -> Dict:
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        return {
            "input": f"User: {question}\nAssistant: Let me solve this step by step.\n",
            "output": answer,
            "task_type": "chain_of_thought"
        }


# =============================================================================
# UNIFIED DATASET MANAGER
# =============================================================================

class DatasetManager:
    """
    Unified manager for all datasets.

    Handles:
    - Dataset discovery and loading
    - Mixing multiple datasets
    - Curriculum-aware sampling
    """

    def __init__(self, cache_dir: str = "./data_cache"):
        self.cache_dir = cache_dir
        self.loaders = {
            "tinystories": TinyStoriesLoader(cache_dir),
            "bfcl": BFCLLoader(cache_dir),
            "nl2bash": NL2BashLoader(cache_dir),
            "glaive_function_calling": GlaiveFunctionCallingLoader(cache_dir),
            "gsm8k": GSM8KLoader(cache_dir),
        }

    def list_datasets(self, task_type: str = None) -> List[DatasetInfo]:
        """List available datasets, optionally filtered by task type."""
        datasets = list(DATASET_REGISTRY.values())

        if task_type:
            datasets = [d for d in datasets if task_type in d.task_types]

        return sorted(datasets, key=lambda x: x.priority)

    def load_dataset(
        self,
        name: str,
        max_samples: int = None,
        split: str = "train"
    ) -> Iterator[Dict]:
        """Load a specific dataset."""
        if name not in self.loaders:
            raise ValueError(f"No loader for dataset: {name}")

        loader = self.loaders[name]

        try:
            return loader.load(split=split, max_samples=max_samples)
        except TypeError:
            # Some loaders don't have split parameter
            return loader.load(max_samples=max_samples)

    def create_mixed_dataset(
        self,
        task_weights: Dict[str, float],
        samples_per_task: int = 1000
    ) -> List[Dict]:
        """
        Create a mixed dataset with specified task weights.

        Args:
            task_weights: Dict mapping task_type to weight
            samples_per_task: Base number of samples per task

        Returns:
            List of training samples
        """
        all_samples = []

        # Map task types to datasets
        task_to_datasets = {
            "text_completion": ["tinystories"],
            "function_call": ["glaive_function_calling", "bfcl"],
            "bash_command": ["nl2bash"],
            "chain_of_thought": ["gsm8k"],
        }

        for task_type, weight in task_weights.items():
            n_samples = int(samples_per_task * weight)
            datasets = task_to_datasets.get(task_type, [])

            for dataset_name in datasets:
                if dataset_name in self.loaders:
                    try:
                        samples = list(self.load_dataset(
                            dataset_name,
                            max_samples=n_samples // len(datasets)
                        ))
                        all_samples.extend(samples)
                        print(f"Loaded {len(samples)} samples from {dataset_name}")
                    except Exception as e:
                        print(f"Warning: Could not load {dataset_name}: {e}")

        random.shuffle(all_samples)
        return all_samples

    def get_recommended_curriculum(self) -> List[Dict]:
        """
        Get recommended curriculum stages with dataset configs.

        Returns list of stage configurations.
        """
        return [
            {
                "stage": 1,
                "name": "Text Coherence",
                "datasets": ["tinystories"],
                "task_weights": {"text_completion": 1.0},
                "samples": 5000
            },
            {
                "stage": 2,
                "name": "Function Calling",
                "datasets": ["tinystories", "glaive_function_calling"],
                "task_weights": {"text_completion": 0.3, "function_call": 0.7},
                "samples": 5000
            },
            {
                "stage": 3,
                "name": "Bash Commands",
                "datasets": ["tinystories", "glaive_function_calling", "nl2bash"],
                "task_weights": {
                    "text_completion": 0.2,
                    "function_call": 0.4,
                    "bash_command": 0.4
                },
                "samples": 5000
            },
            {
                "stage": 4,
                "name": "Reasoning",
                "datasets": ["tinystories", "glaive_function_calling", "nl2bash", "gsm8k"],
                "task_weights": {
                    "text_completion": 0.15,
                    "function_call": 0.3,
                    "bash_command": 0.25,
                    "chain_of_thought": 0.3
                },
                "samples": 5000
            },
        ]


# =============================================================================
# QUICK RECOMMENDATIONS
# =============================================================================

def print_recommendations():
    """Print dataset recommendations for Dragon Hatching training."""
    print("=" * 70)
    print("DATASET RECOMMENDATIONS FOR DRAGON HATCHING")
    print("=" * 70)

    print("\n📚 ESSENTIAL DATASETS (Priority 1):")
    print("-" * 50)
    for name, info in DATASET_REGISTRY.items():
        if info.priority == 1:
            print(f"\n  {info.name}")
            print(f"    Tasks: {', '.join(info.task_types)}")
            print(f"    Size: {info.size}")
            print(f"    URL: {info.url}")
            if info.hf_path:
                print(f"    HF: datasets.load_dataset('{info.hf_path}')")

    print("\n\n📖 RECOMMENDED DATASETS (Priority 2):")
    print("-" * 50)
    for name, info in DATASET_REGISTRY.items():
        if info.priority == 2:
            print(f"\n  {info.name}")
            print(f"    Tasks: {', '.join(info.task_types)}")
            print(f"    Size: {info.size}")

    print("\n\n🎯 RECOMMENDED TRAINING ORDER:")
    print("-" * 50)
    print("""
    Stage 1: TinyStories (text coherence)
    Stage 2: + Glaive Function Calling v2 (113K examples)
    Stage 3: + NL2Bash (10K bash commands)
    Stage 4: + GSM8K (reasoning with CoT)
    Stage 5: + ToolBench (multi-step tool use)
    """)

    print("\n💡 QUICK START:")
    print("-" * 50)
    print("""
    from datasets import load_dataset

    # Essential datasets
    stories = load_dataset("roneneldan/TinyStories", split="train")
    functions = load_dataset("glaiveai/glaive-function-calling-v2")
    math = load_dataset("gsm8k", "main", split="train")
    """)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print_recommendations()

    print("\n\n" + "=" * 70)
    print("TESTING DATASET LOADERS")
    print("=" * 70)

    manager = DatasetManager()

    # Test NL2Bash (can download directly)
    print("\n\nTesting NL2Bash loader...")
    try:
        samples = list(manager.load_dataset("nl2bash", max_samples=5))
        for s in samples[:3]:
            print(f"\n  Input: {s['input'][:60]}...")
            print(f"  Output: {s['output'][:60]}...")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n\nDataset tests complete!")
