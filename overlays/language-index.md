# language-index.md

## Purpose
Index of language-specific configurations and conventions.

## Detected Languages

### Python (Primary)
- Files: *.py
- Config: requirements.txt
- See: overlays/languages/python.md

## Language Detection
```yaml
detected_languages:
  - language: "python"
    confidence: "high"
    signals:
      - "*.py files present"
      - "requirements.txt present"
    primary: true
```

## Language-Specific Resources
| Language | Config File          | Overlay                     |
|----------|----------------------|-----------------------------|
| Python   | overlays/languages/python.md | checks, style, deps  |
| Rust     | overlays/languages/rust.md   | cargo, clippy        |
| JS/TS    | overlays/languages/js-ts.md  | npm, eslint          |
| Go       | overlays/languages/go.md     | go mod, golangci     |

## Multi-Language Projects
If multiple languages detected:
1. Identify primary language (most code)
2. Load all relevant language overlays
3. Apply checks for each language in scope
4. Merge evidence requirements
