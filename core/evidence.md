# evidence.md

## Principle
Every assertion must have evidence. Missing evidence = FAIL.

## Required Evidence Blocks

### Always Required
```yaml
startup_read_log:
  conditional_read_count: <int>
  mandatory_core: [<files read>]
  conditional_reads:
    - file: "<path>"
      trigger: "<why>"

diff_summary:
  files_changed: [<paths>]
  lines_added: <int>
  lines_removed: <int>
  scope_compliant: true|false

verification_results:
  checks_run:
    - check: "<name>"
      command: "<cmd>"
      outcome: "pass|fail|skip"
      output_summary: "<brief>"
  checks_skipped:
    - check: "<name>"
      reason: "TOOL_MISSING|NOT_APPLICABLE"

tool_versions:
  - tool: "<name>"
    version: "<version|TOOL_MISSING>"
```

### Conditional Evidence Blocks

#### When deps touched
```yaml
dependency_change:
  added: [<pkg@version>]
  removed: [<pkg@version>]
  updated:
    - pkg: "<name>"
      from: "<version>"
      to: "<version>"
  license_check: "pass|fail|skip"
  security_advisory_check: "pass|fail|skip"
```

#### When API/contract touched
```yaml
api_change:
  breaking: true|false
  endpoints_added: [<path>]
  endpoints_removed: [<path>]
  endpoints_modified: [<path>]
  schema_changes: "<summary>"
  backward_compatible: true|false
```

#### When tests modified
```yaml
test_change:
  tests_added: <int>
  tests_removed: <int>
  tests_modified: <int>
  coverage_delta: "<+/-X%|unknown>"
```

## Verifier Checklist
1. startup_read_log exists and valid
2. diff_summary matches actual diff
3. verification_results present
4. tool_versions present
5. Conditional blocks present when triggers match
6. No evidence block contains placeholder values
