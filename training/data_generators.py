# Copyright 2025 - Synthetic Data Generators for Dragon Hatching
# Multi-modal training data: text coherence, function calling, tool use, bash

import json
import random
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Any
from enum import Enum
import itertools


# =============================================================================
# DATA TYPES
# =============================================================================

class TaskType(Enum):
    TEXT_COMPLETION = "text_completion"
    FUNCTION_CALL = "function_call"
    TOOL_USE = "tool_use"
    BASH_COMMAND = "bash_command"
    MULTI_TURN = "multi_turn"
    CHAIN_OF_THOUGHT = "chain_of_thought"


@dataclass
class TrainingSample:
    """A single training sample with metadata."""
    input_text: str
    target_text: str
    task_type: TaskType
    metadata: Dict = field(default_factory=dict)

    def to_tokens(self, tokenizer=None) -> Tuple[List[int], List[int]]:
        """Convert to token IDs (byte-level if no tokenizer)."""
        if tokenizer is None:
            # Byte-level tokenization (matches BDH)
            input_ids = list(self.input_text.encode('utf-8'))
            target_ids = list(self.target_text.encode('utf-8'))
        else:
            input_ids = tokenizer.encode(self.input_text)
            target_ids = tokenizer.encode(self.target_text)
        return input_ids, target_ids


# =============================================================================
# FUNCTION DEFINITIONS FOR TRAINING
# =============================================================================

FUNCTION_REGISTRY = {
    "get_weather": {
        "description": "Get current weather for a location",
        "parameters": {
            "location": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
        },
        "required": ["location"]
    },
    "search_web": {
        "description": "Search the web for information",
        "parameters": {
            "query": {"type": "string", "description": "Search query"},
            "num_results": {"type": "integer", "default": 5}
        },
        "required": ["query"]
    },
    "read_file": {
        "description": "Read contents of a file",
        "parameters": {
            "path": {"type": "string", "description": "File path"},
            "encoding": {"type": "string", "default": "utf-8"}
        },
        "required": ["path"]
    },
    "write_file": {
        "description": "Write content to a file",
        "parameters": {
            "path": {"type": "string", "description": "File path"},
            "content": {"type": "string", "description": "Content to write"},
            "mode": {"type": "string", "enum": ["write", "append"], "default": "write"}
        },
        "required": ["path", "content"]
    },
    "execute_code": {
        "description": "Execute code in a sandbox",
        "parameters": {
            "language": {"type": "string", "enum": ["python", "javascript", "bash"]},
            "code": {"type": "string", "description": "Code to execute"},
            "timeout": {"type": "integer", "default": 30}
        },
        "required": ["language", "code"]
    },
    "calculate": {
        "description": "Perform mathematical calculation",
        "parameters": {
            "expression": {"type": "string", "description": "Math expression"},
            "precision": {"type": "integer", "default": 2}
        },
        "required": ["expression"]
    },
    "send_email": {
        "description": "Send an email",
        "parameters": {
            "to": {"type": "string", "description": "Recipient email"},
            "subject": {"type": "string", "description": "Email subject"},
            "body": {"type": "string", "description": "Email body"}
        },
        "required": ["to", "subject", "body"]
    },
    "create_reminder": {
        "description": "Create a reminder",
        "parameters": {
            "message": {"type": "string", "description": "Reminder message"},
            "time": {"type": "string", "description": "When to remind (ISO format)"}
        },
        "required": ["message", "time"]
    }
}

# =============================================================================
# BASH COMMAND PATTERNS
# =============================================================================

BASH_PATTERNS = {
    "file_operations": [
        ("list files in directory", "ls -la {path}"),
        ("find files by name", "find {path} -name '{pattern}'"),
        ("search in files", "grep -r '{pattern}' {path}"),
        ("copy file", "cp {source} {dest}"),
        ("move file", "mv {source} {dest}"),
        ("remove file", "rm {path}"),
        ("create directory", "mkdir -p {path}"),
        ("show file contents", "cat {path}"),
        ("show first lines", "head -n {n} {path}"),
        ("show last lines", "tail -n {n} {path}"),
    ],
    "git_operations": [
        ("check git status", "git status"),
        ("show git log", "git log --oneline -n {n}"),
        ("create branch", "git checkout -b {branch}"),
        ("switch branch", "git checkout {branch}"),
        ("stage changes", "git add {path}"),
        ("commit changes", "git commit -m '{message}'"),
        ("push to remote", "git push origin {branch}"),
        ("pull from remote", "git pull origin {branch}"),
        ("show diff", "git diff {path}"),
        ("clone repository", "git clone {url}"),
    ],
    "system_info": [
        ("show disk usage", "df -h"),
        ("show memory usage", "free -h"),
        ("show running processes", "ps aux"),
        ("show current directory", "pwd"),
        ("show environment variable", "echo ${var}"),
        ("show system info", "uname -a"),
    ],
    "text_processing": [
        ("count lines", "wc -l {path}"),
        ("sort file", "sort {path}"),
        ("unique lines", "sort {path} | uniq"),
        ("replace text", "sed 's/{old}/{new}/g' {path}"),
        ("extract column", "awk '{{print ${col}}}' {path}"),
    ],
    "network": [
        ("download file", "curl -o {output} {url}"),
        ("check connectivity", "ping -c 4 {host}"),
        ("show open ports", "netstat -tulpn"),
    ],
    "python": [
        ("run python script", "python {path}"),
        ("install package", "pip install {package}"),
        ("run pytest", "pytest {path}"),
        ("check python version", "python --version"),
    ]
}

# =============================================================================
# TEXT COHERENCE PATTERNS
# =============================================================================

COHERENCE_TEMPLATES = {
    "continuation": [
        "The quick brown fox {continuation}",
        "In the beginning, there was {continuation}",
        "Scientists have discovered that {continuation}",
        "The main advantage of this approach is {continuation}",
        "To solve this problem, we need to {continuation}",
    ],
    "question_answer": [
        ("What is {topic}?", "{topic} is {definition}"),
        ("How does {thing} work?", "{thing} works by {mechanism}"),
        ("Why is {topic} important?", "{topic} is important because {reason}"),
        ("When should you use {thing}?", "You should use {thing} when {condition}"),
    ],
    "instruction_following": [
        ("Write a {length} {type} about {topic}", "{content}"),
        ("Explain {concept} in simple terms", "{explanation}"),
        ("Summarize the following: {text}", "{summary}"),
        ("Translate to {language}: {text}", "{translation}"),
    ],
    "reasoning": [
        ("If {premise1} and {premise2}, then {conclusion}"),
        ("Given that {fact}, we can conclude {inference}"),
        ("The evidence suggests that {conclusion} because {reasoning}"),
    ]
}

# =============================================================================
# DATA GENERATORS
# =============================================================================

class FunctionCallGenerator:
    """
    Generates synthetic function calling training data.

    Format:
    User: <natural language request>
    Assistant: <function_call>{"name": "func", "arguments": {...}}</function_call>
    """

    def __init__(self, functions: Dict = None):
        self.functions = functions or FUNCTION_REGISTRY

    def generate_request(self, func_name: str) -> Tuple[str, Dict]:
        """Generate a natural language request for a function."""
        func = self.functions[func_name]
        params = func["parameters"]
        required = func.get("required", [])

        # Generate sample arguments
        args = {}
        for param_name, param_info in params.items():
            if param_name in required or random.random() > 0.5:
                args[param_name] = self._generate_param_value(param_name, param_info)

        # Generate natural language request
        request = self._generate_natural_request(func_name, func["description"], args)

        return request, args

    def _generate_param_value(self, name: str, info: Dict) -> Any:
        """Generate a sample value for a parameter."""
        param_type = info.get("type", "string")

        if "enum" in info:
            return random.choice(info["enum"])

        if param_type == "string":
            examples = {
                "location": ["New York", "London", "Tokyo", "Paris", "Berlin"],
                "query": ["latest news", "python tutorial", "machine learning", "weather forecast"],
                "path": ["/home/user/file.txt", "./data/config.json", "../src/main.py"],
                "content": ["Hello, World!", "Sample content here", "Test data"],
                "code": ["print('hello')", "x = 1 + 2", "import numpy as np"],
                "to": ["user@example.com", "admin@test.org"],
                "subject": ["Meeting reminder", "Update required", "Question about project"],
                "body": ["Please review the attached document.", "Looking forward to hearing from you."],
                "message": ["Call mom", "Submit report", "Team meeting"],
                "time": ["2025-01-20T10:00:00", "2025-01-21T14:30:00"],
                "expression": ["2 + 2", "sqrt(16)", "sin(pi/2)", "10 * 5 - 3"],
            }
            return random.choice(examples.get(name, ["sample_value"]))

        if param_type == "integer":
            return random.randint(1, 100)

        return "value"

    def _generate_natural_request(self, func_name: str, description: str, args: Dict) -> str:
        """Generate natural language from function and arguments."""
        templates = {
            "get_weather": [
                f"What's the weather in {args.get('location', 'the city')}?",
                f"Tell me the weather for {args.get('location', 'here')}",
                f"Check the weather in {args.get('location', 'my location')}",
            ],
            "search_web": [
                f"Search for {args.get('query', 'information')}",
                f"Find information about {args.get('query', 'this topic')}",
                f"Look up {args.get('query', 'something')} online",
            ],
            "read_file": [
                f"Read the file at {args.get('path', 'the path')}",
                f"Show me the contents of {args.get('path', 'this file')}",
                f"Open {args.get('path', 'the file')}",
            ],
            "write_file": [
                f"Write '{args.get('content', 'this')}' to {args.get('path', 'a file')}",
                f"Save the following to {args.get('path', 'disk')}: {args.get('content', '')}",
            ],
            "execute_code": [
                f"Run this {args.get('language', 'code')}: {args.get('code', '')}",
                f"Execute the following {args.get('language', '')} code: {args.get('code', '')}",
            ],
            "calculate": [
                f"Calculate {args.get('expression', 'this')}",
                f"What is {args.get('expression', '2+2')}?",
                f"Compute {args.get('expression', 'the result')}",
            ],
            "send_email": [
                f"Send an email to {args.get('to', 'someone')} about {args.get('subject', 'something')}",
                f"Email {args.get('to', 'the recipient')}: {args.get('body', '')}",
            ],
            "create_reminder": [
                f"Remind me to {args.get('message', 'do something')} at {args.get('time', 'later')}",
                f"Set a reminder: {args.get('message', '')}",
            ],
        }

        func_templates = templates.get(func_name, [f"Use {func_name} with {args}"])
        return random.choice(func_templates)

    def generate_sample(self) -> TrainingSample:
        """Generate a complete function calling training sample."""
        func_name = random.choice(list(self.functions.keys()))
        request, args = self.generate_request(func_name)

        # Format function call
        function_call = json.dumps({"name": func_name, "arguments": args})
        target = f"<function_call>{function_call}</function_call>"

        return TrainingSample(
            input_text=f"User: {request}\nAssistant:",
            target_text=target,
            task_type=TaskType.FUNCTION_CALL,
            metadata={"function": func_name, "arguments": args}
        )


class BashCommandGenerator:
    """
    Generates synthetic bash command training data.

    Format:
    User: <natural language request>
    Assistant: ```bash
    <command>
    ```
    """

    def __init__(self, patterns: Dict = None):
        self.patterns = patterns or BASH_PATTERNS

    def _fill_template(self, template: str) -> Tuple[str, str]:
        """Fill a bash command template with sample values."""
        fillers = {
            "path": random.choice(["/home/user", ".", "./src", "/tmp", "../data"]),
            "pattern": random.choice(["*.py", "*.txt", "config*", "test_*"]),
            "source": random.choice(["file.txt", "data.json", "script.py"]),
            "dest": random.choice(["backup/", "/tmp/", "archive/"]),
            "n": str(random.randint(5, 20)),
            "branch": random.choice(["main", "develop", "feature/new-thing"]),
            "message": random.choice(["Fix bug", "Add feature", "Update docs"]),
            "url": "https://github.com/user/repo.git",
            "var": random.choice(["PATH", "HOME", "USER"]),
            "old": random.choice(["foo", "old_name", "TODO"]),
            "new": random.choice(["bar", "new_name", "DONE"]),
            "col": str(random.randint(1, 5)),
            "output": "downloaded_file",
            "host": random.choice(["google.com", "github.com", "localhost"]),
            "package": random.choice(["numpy", "pandas", "requests"]),
        }

        filled = template
        for key, value in fillers.items():
            filled = filled.replace("{" + key + "}", value)

        return filled

    def generate_sample(self) -> TrainingSample:
        """Generate a bash command training sample."""
        category = random.choice(list(self.patterns.keys()))
        description, template = random.choice(self.patterns[category])

        command = self._fill_template(template)

        # Natural language variations
        prefixes = [
            f"How do I {description}?",
            f"I want to {description}",
            f"Can you {description}?",
            f"Help me {description}",
            f"What command will {description}?",
        ]

        request = random.choice(prefixes)
        target = f"```bash\n{command}\n```"

        return TrainingSample(
            input_text=f"User: {request}\nAssistant:",
            target_text=target,
            task_type=TaskType.BASH_COMMAND,
            metadata={"category": category, "command": command}
        )


class ToolUseGenerator:
    """
    Generates multi-step tool use training data.

    Combines function calls with reasoning and intermediate outputs.
    """

    def __init__(self):
        self.func_gen = FunctionCallGenerator()
        self.bash_gen = BashCommandGenerator()

    def generate_tool_chain(self, n_steps: int = 2) -> TrainingSample:
        """Generate a multi-step tool use sequence."""
        steps = []
        context = "User: Help me analyze a Python project.\nAssistant: I'll help you analyze the project step by step.\n\n"

        # Generate sequence of tool uses
        tool_sequence = [
            ("bash", "First, let me list the project structure.", "ls -la"),
            ("bash", "Now let me find all Python files.", "find . -name '*.py'"),
            ("bash", "Let me check the main module.", "cat main.py"),
            ("function", "I'll search for documentation.", {"name": "search_web", "arguments": {"query": "python project structure best practices"}}),
        ]

        selected = random.sample(tool_sequence[:n_steps+1], min(n_steps, len(tool_sequence)))

        output_parts = [context]
        for tool_type, reasoning, tool_data in selected:
            output_parts.append(f"**Step {len(steps)+1}:** {reasoning}\n")

            if tool_type == "bash":
                output_parts.append(f"```bash\n{tool_data}\n```\n")
            else:
                output_parts.append(f"<function_call>{json.dumps(tool_data)}</function_call>\n")

            steps.append((tool_type, tool_data))
            output_parts.append("\n")

        return TrainingSample(
            input_text=context.split("Assistant:")[0] + "Assistant:",
            target_text="".join(output_parts).split("Assistant:")[1],
            task_type=TaskType.TOOL_USE,
            metadata={"steps": steps, "n_steps": len(steps)}
        )

    def generate_sample(self) -> TrainingSample:
        """Generate a tool use sample (single or multi-step)."""
        if random.random() > 0.5:
            return self.generate_tool_chain(n_steps=random.randint(2, 4))
        else:
            # Single tool use
            if random.random() > 0.5:
                return self.func_gen.generate_sample()
            else:
                return self.bash_gen.generate_sample()


class CoherenceGenerator:
    """
    Generates text coherence training data.

    Focuses on:
    - Logical flow
    - Consistent style
    - Proper transitions
    - Topic maintenance
    """

    def __init__(self):
        self.topics = [
            "machine learning", "neural networks", "software engineering",
            "data structures", "algorithms", "distributed systems",
            "database design", "API development", "testing strategies",
            "code optimization", "security practices", "DevOps",
        ]

        self.connectors = [
            "Furthermore,", "Additionally,", "Moreover,",
            "However,", "On the other hand,", "In contrast,",
            "Therefore,", "As a result,", "Consequently,",
            "For example,", "Specifically,", "In particular,",
        ]

    def generate_coherent_paragraph(self, topic: str, n_sentences: int = 4) -> str:
        """Generate a coherent paragraph about a topic."""
        templates = [
            f"{topic.title()} is a fundamental concept in computer science.",
            f"Understanding {topic} is essential for modern software development.",
            f"The key principles of {topic} include several important aspects.",
            f"When working with {topic}, developers should consider various factors.",
            f"Best practices in {topic} have evolved significantly over time.",
            f"The implementation of {topic} requires careful planning and design.",
        ]

        sentences = [random.choice(templates)]

        for i in range(n_sentences - 1):
            connector = random.choice(self.connectors)
            continuation = random.choice([
                f"this enables better system performance.",
                f"this approach provides significant advantages.",
                f"developers can achieve more reliable results.",
                f"the overall architecture becomes more maintainable.",
                f"testing and debugging become more straightforward.",
            ])
            sentences.append(f"{connector} {continuation}")

        return " ".join(sentences)

    def generate_qa_pair(self) -> Tuple[str, str]:
        """Generate a question-answer pair."""
        topic = random.choice(self.topics)

        questions = [
            f"What is {topic}?",
            f"How does {topic} work?",
            f"Why is {topic} important?",
            f"What are the benefits of {topic}?",
            f"How do you implement {topic}?",
        ]

        question = random.choice(questions)
        answer = self.generate_coherent_paragraph(topic, n_sentences=3)

        return question, answer

    def generate_sample(self) -> TrainingSample:
        """Generate a coherence training sample."""
        sample_type = random.choice(["qa", "continuation", "instruction"])

        if sample_type == "qa":
            question, answer = self.generate_qa_pair()
            return TrainingSample(
                input_text=f"User: {question}\nAssistant:",
                target_text=answer,
                task_type=TaskType.TEXT_COMPLETION,
                metadata={"subtype": "qa"}
            )

        elif sample_type == "continuation":
            topic = random.choice(self.topics)
            start = f"Let me explain {topic}. "
            continuation = self.generate_coherent_paragraph(topic, n_sentences=3)
            return TrainingSample(
                input_text=f"User: Explain {topic}.\nAssistant: {start}",
                target_text=continuation,
                task_type=TaskType.TEXT_COMPLETION,
                metadata={"subtype": "continuation"}
            )

        else:  # instruction
            topic = random.choice(self.topics)
            instruction = f"Write a brief explanation of {topic} for beginners."
            response = self.generate_coherent_paragraph(topic, n_sentences=4)
            return TrainingSample(
                input_text=f"User: {instruction}\nAssistant:",
                target_text=response,
                task_type=TaskType.TEXT_COMPLETION,
                metadata={"subtype": "instruction"}
            )


class ChainOfThoughtGenerator:
    """
    Generates chain-of-thought reasoning samples.

    Teaches the model to show its reasoning process.
    """

    def __init__(self):
        self.math_problems = [
            ("If a train travels 60 mph for 2.5 hours, how far does it go?",
             "Let me work through this step by step.\n"
             "1. Speed = 60 mph\n"
             "2. Time = 2.5 hours\n"
             "3. Distance = Speed × Time\n"
             "4. Distance = 60 × 2.5 = 150 miles\n\n"
             "The train travels **150 miles**."),

            ("What is 15% of 80?",
             "Let me calculate this:\n"
             "1. 15% means 15/100 = 0.15\n"
             "2. 0.15 × 80 = 12\n\n"
             "**15% of 80 is 12**."),

            ("If you have 3 boxes with 4 items each, how many items total?",
             "Let me think through this:\n"
             "1. Number of boxes = 3\n"
             "2. Items per box = 4\n"
             "3. Total items = 3 × 4 = 12\n\n"
             "There are **12 items** in total."),
        ]

        self.logic_problems = [
            ("All cats are mammals. Fluffy is a cat. What can we conclude?",
             "Let me reason through this:\n"
             "1. Premise 1: All cats are mammals\n"
             "2. Premise 2: Fluffy is a cat\n"
             "3. Since Fluffy is a cat, and all cats are mammals...\n"
             "4. Conclusion: Fluffy is a mammal\n\n"
             "We can conclude that **Fluffy is a mammal**."),
        ]

        self.code_problems = [
            ("What does this code do: `[x*2 for x in range(5)]`?",
             "Let me trace through this Python list comprehension:\n"
             "1. `range(5)` produces: 0, 1, 2, 3, 4\n"
             "2. For each x, we compute x*2:\n"
             "   - 0*2 = 0\n"
             "   - 1*2 = 2\n"
             "   - 2*2 = 4\n"
             "   - 3*2 = 6\n"
             "   - 4*2 = 8\n"
             "3. Result: [0, 2, 4, 6, 8]\n\n"
             "The code produces **[0, 2, 4, 6, 8]**."),
        ]

    def generate_sample(self) -> TrainingSample:
        """Generate a chain-of-thought sample."""
        all_problems = self.math_problems + self.logic_problems + self.code_problems
        question, answer = random.choice(all_problems)

        return TrainingSample(
            input_text=f"User: {question}\nAssistant:",
            target_text=answer,
            task_type=TaskType.CHAIN_OF_THOUGHT,
            metadata={"requires_reasoning": True}
        )


# =============================================================================
# UNIFIED DATA GENERATOR
# =============================================================================

class UnifiedDataGenerator:
    """
    Unified generator for all training data types.

    Supports weighted sampling across different task types.
    """

    def __init__(
        self,
        weights: Dict[TaskType, float] = None
    ):
        self.generators = {
            TaskType.FUNCTION_CALL: FunctionCallGenerator(),
            TaskType.BASH_COMMAND: BashCommandGenerator(),
            TaskType.TOOL_USE: ToolUseGenerator(),
            TaskType.TEXT_COMPLETION: CoherenceGenerator(),
            TaskType.CHAIN_OF_THOUGHT: ChainOfThoughtGenerator(),
        }

        # Default weights (emphasize tool use and function calling)
        self.weights = weights or {
            TaskType.FUNCTION_CALL: 0.25,
            TaskType.BASH_COMMAND: 0.20,
            TaskType.TOOL_USE: 0.20,
            TaskType.TEXT_COMPLETION: 0.20,
            TaskType.CHAIN_OF_THOUGHT: 0.15,
        }

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}

    def generate_sample(self) -> TrainingSample:
        """Generate a single training sample."""
        task_type = random.choices(
            list(self.weights.keys()),
            weights=list(self.weights.values())
        )[0]

        generator = self.generators[task_type]
        return generator.generate_sample()

    def generate_batch(self, batch_size: int) -> List[TrainingSample]:
        """Generate a batch of training samples."""
        return [self.generate_sample() for _ in range(batch_size)]

    def generate_dataset(self, n_samples: int) -> List[TrainingSample]:
        """Generate a full dataset."""
        return self.generate_batch(n_samples)


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("Testing Data Generators")
    print("=" * 60)

    gen = UnifiedDataGenerator()

    # Generate samples of each type
    for task_type in TaskType:
        print(f"\n{task_type.value.upper()}")
        print("-" * 40)

        if task_type in gen.generators:
            sample = gen.generators[task_type].generate_sample()
            print(f"Input:\n{sample.input_text[:200]}...")
            print(f"\nTarget:\n{sample.target_text[:200]}...")

    # Generate mixed batch
    print("\n" + "=" * 60)
    print("MIXED BATCH (10 samples)")
    print("-" * 40)

    batch = gen.generate_batch(10)
    for sample in batch:
        print(f"  [{sample.task_type.value}] {sample.input_text[:50]}...")

    print("\nData generator tests complete!")
