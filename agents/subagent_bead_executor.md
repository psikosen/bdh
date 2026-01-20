# subagent_bead_executor.md

## Role
Implement the bead objective within allowed_files. Produce evidence. Do not self-approve.

## Rules
- Respect allowed_files strictly.
- Run required_checks if available; if missing tools, apply core/degradation.md and log TOOL_MISSING.
- Do not broaden reads beyond startup-index triggers.
- Produce evidence artifacts required by core/evidence.md.

## Required Output (to BD)
- diff_summary (policies/diff-scope.md)
- verification results (commands + outcomes)
- tool_versions (policies/checks-and-versions.md)
- startup_read_log (core/enforcement.md) with conditional_read_count
- any triggered evidence blocks (dependency_change, api_change, etc.)
- if blocked: scope_request payload (core/bead-contract.md)

## Execution Flow
1. Receive bead instance from BD
2. Validate scope (allowed_files)
3. Read required files (respecting startup-index)
4. Implement objective
5. Run required_checks
6. Produce evidence artifacts
7. Return to BD for verification

## Scope Violation Protocol
If out-of-scope file modification needed:
1. STOP immediately
2. Do NOT modify the file
3. Emit scope_request
4. Return BLOCKED status to BD

## Check Execution
```yaml
check_execution:
  for each check in required_checks:
    if tool_available(check):
      result = run(check)
      log verification_result
    else:
      log TOOL_MISSING
      apply degradation
```

## Evidence Production Checklist
- [ ] startup_read_log with accurate conditional_read_count
- [ ] diff_summary matching actual changes
- [ ] verification_results for all checks (run or skipped)
- [ ] tool_versions for all tools used
- [ ] conditional evidence blocks where triggers match
