# python.md

## Language: Python

## Package Management
- requirements.txt (detected)
- pyproject.toml (if present)
- setup.py (legacy)

## Default Checks
```yaml
python_checks:
  test:
    commands:
      - "pytest"
      - "python -m pytest"
    fallback: "python -m unittest discover"

  lint:
    commands:
      - "ruff check ."
      - "flake8"
      - "pylint"
    preference: "ruff > flake8 > pylint"

  format:
    commands:
      - "ruff format --check ."
      - "black --check ."
    preference: "ruff > black"

  typecheck:
    commands:
      - "mypy ."
      - "pyright"
    optional: true
```

## Style Guidelines
- PEP 8 compliance
- Type hints encouraged (PEP 484)
- Docstrings for public APIs (PEP 257)
- Max line length: 88 (black default) or 79 (PEP 8)

## Dependency Management
```yaml
dep_files:
  - requirements.txt
  - requirements-dev.txt
  - pyproject.toml [project.dependencies]
  - setup.py [install_requires]

security_check:
  - "pip-audit"
  - "safety check"
```

## Virtual Environment
- Prefer venv or virtualenv
- Never install to system Python
- Document Python version requirement

## Common Patterns
```yaml
patterns:
  dataclass_config: "@dataclasses.dataclass for config"
  type_hints: "def func(x: int) -> str:"
  context_manager: "with open() as f:"
  f_strings: 'f"value: {var}"'
```

## Testing Conventions
- Test files: test_*.py or *_test.py
- Test directory: tests/
- Fixtures in conftest.py
- pytest markers for categorization
