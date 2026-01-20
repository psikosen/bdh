# Dataset Recommendations for Dragon Hatching

This document outlines recommended datasets for training the Dragon Hatching model across different capabilities.

## Quick Start

```bash
pip install datasets

# Download essential datasets
python -c "
from datasets import load_dataset

# Text coherence
stories = load_dataset('roneneldan/TinyStories', split='train[:10000]')

# Function calling
functions = load_dataset('glaiveai/glaive-function-calling-v2', split='train')

# Chain of thought
math = load_dataset('gsm8k', 'main', split='train')
"
```

---

## Training Stages & Datasets

### Stage 1: Text Coherence (~2000 iterations)

| Dataset | Size | Description | Link |
|---------|------|-------------|------|
| **TinyStories** | 500MB | Synthetic coherent stories | [HuggingFace](https://huggingface.co/datasets/roneneldan/TinyStories) |
| WikiText-103 | 500MB | Wikipedia articles | [HuggingFace](https://huggingface.co/datasets/wikitext) |

**Why these?** TinyStories teaches narrative coherence with simple vocabulary. Perfect for establishing base language modeling.

```python
from datasets import load_dataset
stories = load_dataset("roneneldan/TinyStories", split="train")
```

---

### Stage 2: Function Calling (~3000 iterations)

| Dataset | Size | Description | Link |
|---------|------|-------------|------|
| **Glaive Function Calling v2** | 150MB | 113K function calling conversations | [HuggingFace](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) |
| **BFCL** | 50MB | Berkeley benchmark (gold standard) | [HuggingFace](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) |
| Gorilla APIBench | 100MB | Real API usage patterns | [GitHub](https://github.com/ShishirPatil/gorilla) |

**Why these?** Glaive provides volume, BFCL provides quality evaluation data with diverse function schemas.

```python
from datasets import load_dataset
functions = load_dataset("glaiveai/glaive-function-calling-v2")
bfcl = load_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard")
```

---

### Stage 3: Bash Commands (~3000 iterations)

| Dataset | Size | Description | Link |
|---------|------|-------------|------|
| **NL2Bash** | 5MB | 10K bash one-liners | [GitHub](https://github.com/TellinaTool/nl2bash) |
| unix-commands | 10MB | Unix command examples | [HuggingFace](https://huggingface.co/datasets/harpomaxx/unix-commands) |
| bash-commands-dataset | 5MB | NL to bash pairs | [HuggingFace](https://huggingface.co/datasets/aelhalili/bash-commands-dataset) |

**Why these?** NL2Bash is the definitive dataset for natural language → bash translation with 102 unique utilities.

```python
# NL2Bash (manual download)
# wget https://github.com/TellinaTool/nl2bash/raw/master/data/bash/all.nl
# wget https://github.com/TellinaTool/nl2bash/raw/master/data/bash/all.cm

# Or use our loader
from training.datasets import NL2BashLoader
loader = NL2BashLoader()
samples = list(loader.load(max_samples=1000))
```

---

### Stage 4: Chain of Thought / Reasoning (~2000 iterations)

| Dataset | Size | Description | Link |
|---------|------|-------------|------|
| **GSM8K** | 10MB | Grade school math with solutions | [HuggingFace](https://huggingface.co/datasets/gsm8k) |
| AQuA-RAT | 20MB | Math problems with rationales | [HuggingFace](https://huggingface.co/datasets/aqua_rat) |
| Code Alpaca | 20MB | Code instruction following | [HuggingFace](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) |

**Why these?** GSM8K teaches step-by-step reasoning with clear intermediate steps.

```python
from datasets import load_dataset
gsm8k = load_dataset("gsm8k", "main", split="train")
```

---

### Stage 5: Multi-Step Tool Use (Advanced)

| Dataset | Size | Description | Link |
|---------|------|-------------|------|
| **ToolBench** | 500MB | Multi-step tool sequences | [GitHub](https://github.com/OpenBMB/ToolBench) |
| AgentInstruct | 100MB | Agent interaction patterns | [HuggingFace](https://huggingface.co/datasets/THUDM/AgentInstruct) |

**Why these?** ToolBench provides real multi-step tool usage patterns critical for agent behavior.

---

## Dataset Sizes by Training Budget

### Small Budget (~10K samples)
```
TinyStories: 3000 samples
Glaive Functions: 3000 samples
NL2Bash: 2000 samples
GSM8K: 2000 samples
```

### Medium Budget (~50K samples)
```
TinyStories: 15000 samples
Glaive Functions: 15000 samples
BFCL: 2000 samples
NL2Bash: 8000 samples
GSM8K: 7500 samples
Code Alpaca: 2500 samples
```

### Large Budget (~200K+ samples)
```
TinyStories: 50000 samples
OpenWebText subset: 30000 samples
Glaive Functions: 50000 samples
BFCL + APIBench: 10000 samples
NL2Bash + unix-commands: 15000 samples
GSM8K + AQuA: 15000 samples
ToolBench: 20000 samples
OpenCodeInstruct subset: 10000 samples
```

---

## Data Format

All datasets are converted to this unified format:

```json
{
  "input": "User: What is the weather in Tokyo?\nAssistant:",
  "output": "<function_call>{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Tokyo\"}}</function_call>",
  "task_type": "function_call"
}
```

### Task Types
- `text_completion` - Continue coherent text
- `function_call` - Generate function call JSON
- `bash_command` - Generate bash in code block
- `chain_of_thought` - Step-by-step reasoning
- `tool_use` - Multi-step tool sequences

---

## Evaluation Datasets

For evaluation (not training):

| Benchmark | Purpose | Link |
|-----------|---------|------|
| BFCL v4 | Function calling accuracy | [Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) |
| WebArena | Agent web tasks | [GitHub](https://webarena.dev/) |
| HumanEval | Code generation | [GitHub](https://github.com/openai/human-eval) |
| MMLU | General knowledge | [HuggingFace](https://huggingface.co/datasets/cais/mmlu) |

---

## Usage with Dragon Hatching

```python
from training.datasets import DatasetManager

manager = DatasetManager(cache_dir="./data")

# Get recommended curriculum
curriculum = manager.get_recommended_curriculum()

# Load mixed dataset for a stage
mixed_data = manager.create_mixed_dataset(
    task_weights={
        "text_completion": 0.2,
        "function_call": 0.4,
        "bash_command": 0.3,
        "chain_of_thought": 0.1
    },
    samples_per_task=2000
)

print(f"Created {len(mixed_data)} training samples")
```

---

## Sources

- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
- [Gorilla: Training LLMs for Function Calls](https://github.com/ShishirPatil/gorilla)
- [NL2Bash Paper](https://arxiv.org/abs/1802.08979)
- [OpenCodeInstruct](https://arxiv.org/abs/2504.04030)
- [mlabonne/llm-datasets](https://github.com/mlabonne/llm-datasets) - Curated list
