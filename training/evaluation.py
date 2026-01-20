# Copyright 2025 - Evaluation Metrics for Dragon Hatching
# Coherence, function calling accuracy, tool use, and bash correctness

import json
import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter
import torch
import torch.nn.functional as F


# =============================================================================
# COHERENCE METRICS
# =============================================================================

@dataclass
class CoherenceMetrics:
    """Collection of text coherence metrics."""
    perplexity: float = 0.0
    repetition_ratio: float = 0.0
    sentence_coherence: float = 0.0
    topic_consistency: float = 0.0
    overall_score: float = 0.0


class CoherenceEvaluator:
    """
    Evaluates text coherence using multiple metrics.

    Metrics:
    1. Perplexity (lower is better)
    2. Repetition detection (less repetition is better)
    3. Sentence-level coherence (semantic similarity between adjacent sentences)
    4. Topic consistency (stable topic throughout)
    """

    def __init__(self, model=None):
        self.model = model

    def compute_perplexity(self, text: str, model=None) -> float:
        """
        Compute perplexity of generated text.

        Perplexity = exp(average negative log-likelihood)
        Lower perplexity = more coherent/predictable text
        """
        model = model or self.model
        if model is None:
            return 0.0

        # Convert to tokens (byte-level)
        tokens = torch.tensor([list(text.encode('utf-8'))])

        with torch.no_grad():
            logits, loss, _ = model(tokens[:, :-1], tokens[:, 1:])
            if loss is not None:
                return math.exp(loss.item())

        return 0.0

    def compute_repetition_ratio(self, text: str, n: int = 3) -> float:
        """
        Compute n-gram repetition ratio.

        Higher ratio = more repetition = less coherent
        """
        words = text.lower().split()
        if len(words) < n:
            return 0.0

        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        unique_ngrams = set(ngrams)

        if len(ngrams) == 0:
            return 0.0

        # Ratio of repeated n-grams
        repetition = 1.0 - len(unique_ngrams) / len(ngrams)
        return repetition

    def compute_sentence_coherence(self, text: str) -> float:
        """
        Compute coherence between adjacent sentences.

        Uses word overlap as a proxy for semantic similarity.
        Higher overlap = better flow between sentences.
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            return 1.0

        coherence_scores = []
        for i in range(len(sentences) - 1):
            words1 = set(sentences[i].lower().split())
            words2 = set(sentences[i+1].lower().split())

            # Jaccard similarity
            if len(words1 | words2) == 0:
                continue

            similarity = len(words1 & words2) / len(words1 | words2)
            coherence_scores.append(similarity)

        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0

    def compute_topic_consistency(self, text: str) -> float:
        """
        Compute topic consistency throughout the text.

        Checks if key terms appear throughout (not just at start/end).
        """
        words = text.lower().split()
        if len(words) < 20:
            return 1.0

        # Find content words (longer words, likely topic-related)
        content_words = [w for w in words if len(w) > 5]
        if not content_words:
            return 0.5

        # Count occurrences
        word_counts = Counter(content_words)
        top_words = [w for w, c in word_counts.most_common(5) if c > 1]

        if not top_words:
            return 0.5

        # Check distribution across text
        thirds = [words[:len(words)//3], words[len(words)//3:2*len(words)//3], words[2*len(words)//3:]]

        consistency_scores = []
        for word in top_words:
            present_in = sum(1 for third in thirds if word in [w.lower() for w in third])
            consistency_scores.append(present_in / 3.0)

        return sum(consistency_scores) / len(consistency_scores)

    def evaluate(self, text: str, model=None) -> CoherenceMetrics:
        """Compute all coherence metrics."""
        perplexity = self.compute_perplexity(text, model)
        repetition = self.compute_repetition_ratio(text)
        sentence_coh = self.compute_sentence_coherence(text)
        topic_cons = self.compute_topic_consistency(text)

        # Overall score (weighted combination)
        # Normalize perplexity (assume good perplexity < 100)
        ppl_score = max(0, 1 - perplexity / 100) if perplexity > 0 else 0.5

        overall = (
            0.3 * ppl_score +
            0.2 * (1 - repetition) +  # Less repetition is better
            0.25 * sentence_coh +
            0.25 * topic_cons
        )

        return CoherenceMetrics(
            perplexity=perplexity,
            repetition_ratio=repetition,
            sentence_coherence=sentence_coh,
            topic_consistency=topic_cons,
            overall_score=overall
        )


# =============================================================================
# FUNCTION CALLING METRICS
# =============================================================================

@dataclass
class FunctionCallMetrics:
    """Metrics for function calling accuracy."""
    format_valid: bool = False
    function_correct: bool = False
    arguments_correct: bool = False
    argument_types_valid: bool = False
    required_args_present: bool = False
    overall_score: float = 0.0


class FunctionCallEvaluator:
    """
    Evaluates function calling output.

    Checks:
    1. Valid JSON format
    2. Correct function name
    3. Correct argument names
    4. Valid argument types
    5. All required arguments present
    """

    def __init__(self, function_registry: Dict = None):
        from data_generators import FUNCTION_REGISTRY
        self.registry = function_registry or FUNCTION_REGISTRY

    def extract_function_call(self, text: str) -> Optional[Dict]:
        """Extract function call from text."""
        # Look for <function_call>...</function_call> tags
        match = re.search(r'<function_call>(.*?)</function_call>', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return None
        return None

    def evaluate(
        self,
        generated: str,
        expected_function: str = None,
        expected_args: Dict = None
    ) -> FunctionCallMetrics:
        """Evaluate function call output."""
        metrics = FunctionCallMetrics()

        # Extract function call
        call = self.extract_function_call(generated)
        if call is None:
            return metrics

        metrics.format_valid = True

        # Check function name
        func_name = call.get('name', '')
        if expected_function:
            metrics.function_correct = (func_name == expected_function)
        else:
            metrics.function_correct = func_name in self.registry

        # Check arguments
        args = call.get('arguments', {})
        if expected_args:
            metrics.arguments_correct = (args == expected_args)
        else:
            metrics.arguments_correct = isinstance(args, dict)

        # Check argument types against registry
        if func_name in self.registry:
            func_def = self.registry[func_name]
            param_defs = func_def.get('parameters', {})
            required = func_def.get('required', [])

            # Check required args
            metrics.required_args_present = all(r in args for r in required)

            # Check types
            type_valid = True
            for arg_name, arg_value in args.items():
                if arg_name in param_defs:
                    expected_type = param_defs[arg_name].get('type', 'string')
                    if expected_type == 'string' and not isinstance(arg_value, str):
                        type_valid = False
                    elif expected_type == 'integer' and not isinstance(arg_value, int):
                        type_valid = False
                    elif 'enum' in param_defs[arg_name]:
                        if arg_value not in param_defs[arg_name]['enum']:
                            type_valid = False
            metrics.argument_types_valid = type_valid

        # Overall score
        scores = [
            metrics.format_valid,
            metrics.function_correct,
            metrics.arguments_correct,
            metrics.argument_types_valid,
            metrics.required_args_present
        ]
        metrics.overall_score = sum(scores) / len(scores)

        return metrics


# =============================================================================
# BASH COMMAND METRICS
# =============================================================================

@dataclass
class BashMetrics:
    """Metrics for bash command evaluation."""
    format_valid: bool = False
    command_exists: bool = False
    syntax_valid: bool = False
    flags_valid: bool = False
    safety_check: bool = True  # True if safe
    overall_score: float = 0.0


class BashEvaluator:
    """
    Evaluates bash command output.

    Checks:
    1. Valid code block format
    2. Command exists
    3. Basic syntax validity
    4. Flag validity
    5. Safety (no dangerous commands)
    """

    KNOWN_COMMANDS = {
        'ls', 'cd', 'pwd', 'mkdir', 'rm', 'cp', 'mv', 'cat', 'head', 'tail',
        'grep', 'find', 'sed', 'awk', 'sort', 'uniq', 'wc', 'echo', 'touch',
        'chmod', 'chown', 'tar', 'gzip', 'gunzip', 'zip', 'unzip',
        'git', 'npm', 'pip', 'python', 'node', 'curl', 'wget',
        'docker', 'kubectl', 'make', 'cmake', 'cargo', 'go',
        'ps', 'kill', 'top', 'df', 'du', 'free', 'netstat', 'ping',
        'ssh', 'scp', 'rsync', 'date', 'cal', 'man', 'which', 'whereis',
        'export', 'env', 'source', 'alias', 'history', 'clear',
        'apt', 'apt-get', 'yum', 'brew', 'pacman',
        'systemctl', 'service', 'journalctl',
        'pytest', 'jest', 'mocha', 'rspec',
    }

    DANGEROUS_PATTERNS = [
        r'rm\s+-rf\s+/',  # rm -rf /
        r':\(\)\s*\{\s*:\|:&\s*\}\s*;',  # Fork bomb
        r'>\s*/dev/sd[a-z]',  # Write to disk device
        r'mkfs\.',  # Format filesystem
        r'dd\s+if=.*of=/dev',  # dd to device
        r'chmod\s+-R\s+777\s+/',  # chmod 777 root
        r'\$\(.*\)',  # Command substitution (can be risky)
    ]

    def extract_bash_command(self, text: str) -> Optional[str]:
        """Extract bash command from text."""
        # Look for ```bash ... ```
        match = re.search(r'```bash\n(.*?)\n```', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Also try ``` ... ```
        match = re.search(r'```\n(.*?)\n```', text, re.DOTALL)
        if match:
            return match.group(1).strip()

        return None

    def check_command_exists(self, command: str) -> bool:
        """Check if the base command is known."""
        # Extract base command (first word)
        parts = command.split()
        if not parts:
            return False

        base_cmd = parts[0]
        # Handle sudo
        if base_cmd == 'sudo' and len(parts) > 1:
            base_cmd = parts[1]

        return base_cmd in self.KNOWN_COMMANDS

    def check_syntax(self, command: str) -> bool:
        """Basic syntax validation."""
        # Check for unmatched quotes
        single_quotes = command.count("'")
        double_quotes = command.count('"')

        if single_quotes % 2 != 0:
            return False
        if double_quotes % 2 != 0:
            return False

        # Check for unmatched parentheses
        if command.count('(') != command.count(')'):
            return False
        if command.count('[') != command.count(']'):
            return False
        if command.count('{') != command.count('}'):
            return False

        return True

    def check_safety(self, command: str) -> bool:
        """Check for dangerous patterns."""
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command):
                return False
        return True

    def evaluate(self, generated: str, expected_command: str = None) -> BashMetrics:
        """Evaluate bash command output."""
        metrics = BashMetrics()

        # Extract command
        command = self.extract_bash_command(generated)
        if command is None:
            return metrics

        metrics.format_valid = True
        metrics.command_exists = self.check_command_exists(command)
        metrics.syntax_valid = self.check_syntax(command)
        metrics.safety_check = self.check_safety(command)

        # Check flags (basic: flags should start with -)
        parts = command.split()
        flags = [p for p in parts if p.startswith('-')]
        metrics.flags_valid = all(re.match(r'^-{1,2}[a-zA-Z]', f) for f in flags) if flags else True

        # Overall score
        scores = [
            metrics.format_valid,
            metrics.command_exists,
            metrics.syntax_valid,
            metrics.flags_valid,
            metrics.safety_check
        ]
        metrics.overall_score = sum(scores) / len(scores)

        return metrics


# =============================================================================
# TOOL USE METRICS
# =============================================================================

@dataclass
class ToolUseMetrics:
    """Metrics for multi-step tool use."""
    n_tools_used: int = 0
    tools_valid: float = 0.0  # Fraction of valid tool calls
    logical_ordering: bool = True
    reasoning_present: bool = False
    task_completion: float = 0.0
    overall_score: float = 0.0


class ToolUseEvaluator:
    """Evaluates multi-step tool use sequences."""

    def __init__(self):
        self.func_eval = FunctionCallEvaluator()
        self.bash_eval = BashEvaluator()

    def extract_tools(self, text: str) -> List[Tuple[str, str]]:
        """Extract all tool uses from text."""
        tools = []

        # Find function calls
        func_matches = re.finditer(r'<function_call>(.*?)</function_call>', text, re.DOTALL)
        for match in func_matches:
            tools.append(('function', match.group(1)))

        # Find bash commands
        bash_matches = re.finditer(r'```bash\n(.*?)\n```', text, re.DOTALL)
        for match in bash_matches:
            tools.append(('bash', match.group(1)))

        return tools

    def evaluate(self, generated: str, expected_tools: List = None) -> ToolUseMetrics:
        """Evaluate tool use sequence."""
        metrics = ToolUseMetrics()

        tools = self.extract_tools(generated)
        metrics.n_tools_used = len(tools)

        if len(tools) == 0:
            return metrics

        # Evaluate each tool
        valid_tools = 0
        for tool_type, tool_content in tools:
            if tool_type == 'function':
                func_metrics = self.func_eval.evaluate(f'<function_call>{tool_content}</function_call>')
                if func_metrics.format_valid and func_metrics.function_correct:
                    valid_tools += 1
            else:  # bash
                bash_metrics = self.bash_eval.evaluate(f'```bash\n{tool_content}\n```')
                if bash_metrics.format_valid and bash_metrics.command_exists:
                    valid_tools += 1

        metrics.tools_valid = valid_tools / len(tools)

        # Check for reasoning (look for step indicators, explanations)
        reasoning_indicators = ['step', 'first', 'then', 'next', 'because', 'since', 'therefore']
        metrics.reasoning_present = any(ind in generated.lower() for ind in reasoning_indicators)

        # Estimate task completion (heuristic: more valid tools + reasoning = better)
        metrics.task_completion = (
            0.4 * metrics.tools_valid +
            0.3 * min(1.0, metrics.n_tools_used / 3) +  # Having 3+ tools is good
            0.3 * float(metrics.reasoning_present)
        )

        # Overall score
        metrics.overall_score = (
            0.3 * metrics.tools_valid +
            0.2 * float(metrics.logical_ordering) +
            0.2 * float(metrics.reasoning_present) +
            0.3 * metrics.task_completion
        )

        return metrics


# =============================================================================
# UNIFIED EVALUATOR
# =============================================================================

@dataclass
class UnifiedMetrics:
    """Combined metrics across all task types."""
    coherence: CoherenceMetrics = field(default_factory=CoherenceMetrics)
    function_call: FunctionCallMetrics = field(default_factory=FunctionCallMetrics)
    bash: BashMetrics = field(default_factory=BashMetrics)
    tool_use: ToolUseMetrics = field(default_factory=ToolUseMetrics)
    overall_score: float = 0.0


class UnifiedEvaluator:
    """Unified evaluator for all task types."""

    def __init__(self, model=None):
        self.coherence_eval = CoherenceEvaluator(model)
        self.func_eval = FunctionCallEvaluator()
        self.bash_eval = BashEvaluator()
        self.tool_eval = ToolUseEvaluator()

    def evaluate(
        self,
        generated: str,
        task_type: str = None,
        model=None,
        **kwargs
    ) -> UnifiedMetrics:
        """Evaluate generated text across all metrics."""
        metrics = UnifiedMetrics()

        # Always compute coherence
        metrics.coherence = self.coherence_eval.evaluate(generated, model)

        # Task-specific evaluation
        if task_type == 'function_call' or '<function_call>' in generated:
            metrics.function_call = self.func_eval.evaluate(
                generated,
                kwargs.get('expected_function'),
                kwargs.get('expected_args')
            )

        if task_type == 'bash' or '```bash' in generated:
            metrics.bash = self.bash_eval.evaluate(
                generated,
                kwargs.get('expected_command')
            )

        if task_type == 'tool_use' or (
            '<function_call>' in generated and '```bash' in generated
        ):
            metrics.tool_use = self.tool_eval.evaluate(
                generated,
                kwargs.get('expected_tools')
            )

        # Compute overall
        scores = [
            metrics.coherence.overall_score,
            metrics.function_call.overall_score,
            metrics.bash.overall_score,
            metrics.tool_use.overall_score
        ]
        # Only count non-zero scores
        active_scores = [s for s in scores if s > 0]
        metrics.overall_score = sum(active_scores) / len(active_scores) if active_scores else 0.0

        return metrics


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("Testing Evaluation Metrics")
    print("=" * 60)

    # Test coherence
    print("\n1. COHERENCE EVALUATION")
    print("-" * 40)
    coherence_eval = CoherenceEvaluator()

    good_text = """
    Machine learning is a fundamental concept in computer science.
    Furthermore, this enables better system performance.
    Additionally, the overall architecture becomes more maintainable.
    Therefore, testing and debugging become more straightforward.
    """

    bad_text = """
    Machine learning machine learning machine learning.
    Random words here there everywhere.
    No connection between sentences at all.
    Purple elephant dancing on the moon.
    """

    good_metrics = coherence_eval.evaluate(good_text)
    bad_metrics = coherence_eval.evaluate(bad_text)

    print(f"Good text: coherence={good_metrics.overall_score:.2f}, "
          f"repetition={good_metrics.repetition_ratio:.2f}")
    print(f"Bad text: coherence={bad_metrics.overall_score:.2f}, "
          f"repetition={bad_metrics.repetition_ratio:.2f}")

    # Test function calling
    print("\n2. FUNCTION CALL EVALUATION")
    print("-" * 40)
    func_eval = FunctionCallEvaluator()

    good_func = '<function_call>{"name": "get_weather", "arguments": {"location": "New York"}}</function_call>'
    bad_func = '<function_call>{"name": "invalid_func", "args": "wrong"}</function_call>'

    good_func_metrics = func_eval.evaluate(good_func, 'get_weather')
    bad_func_metrics = func_eval.evaluate(bad_func)

    print(f"Good function call: score={good_func_metrics.overall_score:.2f}")
    print(f"Bad function call: score={bad_func_metrics.overall_score:.2f}")

    # Test bash
    print("\n3. BASH EVALUATION")
    print("-" * 40)
    bash_eval = BashEvaluator()

    good_bash = '```bash\nls -la /home/user\n```'
    bad_bash = '```bash\nrm -rf /\n```'

    good_bash_metrics = bash_eval.evaluate(good_bash)
    bad_bash_metrics = bash_eval.evaluate(bad_bash)

    print(f"Good bash: score={good_bash_metrics.overall_score:.2f}, safe={good_bash_metrics.safety_check}")
    print(f"Bad bash: score={bad_bash_metrics.overall_score:.2f}, safe={bad_bash_metrics.safety_check}")

    # Test tool use
    print("\n4. TOOL USE EVALUATION")
    print("-" * 40)
    tool_eval = ToolUseEvaluator()

    multi_tool = """
    **Step 1:** First, let me list the files.
    ```bash
    ls -la
    ```

    **Step 2:** Now, let me search for more information.
    <function_call>{"name": "search_web", "arguments": {"query": "python best practices"}}</function_call>
    """

    tool_metrics = tool_eval.evaluate(multi_tool)
    print(f"Multi-tool: n_tools={tool_metrics.n_tools_used}, "
          f"valid={tool_metrics.tools_valid:.2f}, "
          f"reasoning={tool_metrics.reasoning_present}")

    print("\nEvaluation tests complete!")
