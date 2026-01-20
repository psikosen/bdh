# degradation.md

## Degradation Levels

### FULL
- All required tools available
- All checks can run
- Normal operation

### PARTIAL
- Some tools missing
- Critical checks available (at least lint OR test)
- Proceed with reduced confidence

### MINIMAL
- Most tools missing
- Only basic validation possible
- Requires extra caution

## Tool Categories

### Critical
- Test runner (pytest, jest, cargo test, etc.)
- Linter (eslint, clippy, flake8, etc.)

### Important
- Formatter (prettier, black, rustfmt, etc.)
- Type checker (mypy, tsc, etc.)

### Optional
- Coverage tools
- Security scanners
- Documentation generators

## Degradation Rules

### Determining Level
```
if all critical + all important available:
  level = FULL
elif any critical available:
  level = PARTIAL
else:
  level = MINIMAL
```

### Risk × Degradation Matrix
| Risk   | FULL | PARTIAL | MINIMAL |
|--------|------|---------|---------|
| LOW    | OK   | OK      | OK      |
| MEDIUM | OK   | OK      | WARN    |
| HIGH   | OK   | WARN    | BLOCK   |

- OK: Proceed normally
- WARN: Proceed with checkpoint advisory
- BLOCK: Cannot proceed; checkpoint_required

## Evidence Requirements
```yaml
degradation_status:
  level: "full|partial|minimal"
  tools_available:
    - tool: "<name>"
      status: "available|missing"
  missing_checks:
    - check: "<name>"
      reason: "TOOL_MISSING"
```

## Recovery
When tools become available:
1. Re-run skipped checks
2. Update degradation_status
3. Re-verify if risk warranted WARN/BLOCK
