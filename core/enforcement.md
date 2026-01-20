# enforcement.md

## REQUIRED
- Follow core/startup-index.md exactly.
- Do not read/use non-core files unless trigger fired.
- Referencing non-triggered policy/file as justification = FAIL.
- If conditional_read_count > 6 without an approved checkpoint = FAIL.

## EVIDENCE (required)
```yaml
startup_read_log:
  conditional_read_count: <int>
  mandatory_core: [ ... ]
  conditional_reads:
    - file: "<file>"
      trigger: "<why>"
```

## VERIFIER
- Validate startup_read_log exists
- Validate each conditional read has a matching trigger
- Fail on violations; if uncertain → checkpoint_required
