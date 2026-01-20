# repo-conventions.md

## Purpose
Document repository-specific conventions and patterns.

## Project: Baby Dragon Hatchling (BDH)

### Structure
```
bdh/
├── bdh.py          # Main model implementation
├── train.py        # Training script
├── requirements.txt # Python dependencies
├── figs/           # Documentation figures
├── core/           # Beads system core
├── agents/         # Beads system agents
├── policies/       # Beads system policies
└── overlays/       # Beads system overlays
```

### Language
- Primary: Python 3.x
- Framework: PyTorch

### Code Style
- Follow PEP 8
- Use type hints where practical
- Prefer dataclasses for configuration

### Testing
- Test runner: pytest (if available)
- Test location: tests/ (when created)

### Dependencies
- Managed via requirements.txt
- Core deps: torch

### Commit Conventions
- Use imperative mood
- Keep subject line under 72 chars
- Reference issues when applicable

### Documentation
- README.md for project overview
- Code comments for complex logic
- Docstrings for public APIs

## Detected Patterns
- Single-file model architecture
- Dataclass-based configuration
- Clean separation of model and training

## Recommendations
- Add type hints to public functions
- Consider adding tests for model components
- Document hyperparameter choices
