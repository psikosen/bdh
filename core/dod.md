# dod.md (Definition of Done)

## Base Definition of Done
A bead is DONE when:

1. **Objective Met**
   - bead.objective accomplished
   - expected_outputs produced
   - No stop_conditions violated

2. **Scope Respected**
   - All changes within allowed_files
   - No out-of-scope modifications
   - diff_summary.scope_compliant = true

3. **Checks Passed**
   - All required_checks executed (or properly degraded)
   - No check failures (unless documented exceptions)
   - verification_results complete

4. **Evidence Complete**
   - All required evidence blocks present
   - All conditional evidence blocks present (when triggered)
   - No placeholder values

5. **Verifier Approved**
   - Adversarial verifier returns PASS
   - No unresolved findings with severity=high

## Custom DoD
Beads may specify custom_dod to add requirements:
```yaml
custom_dod:
  - requirement: "<description>"
    evidence_key: "<what proves it>"
```

Custom DoD is ADDITIVE only - cannot remove base requirements.

## DoD by Risk Level

### LOW
- Base DoD
- lint + fmt checks

### MEDIUM
- Base DoD
- lint + fmt + test checks
- No test regressions

### HIGH
- Base DoD
- All available checks
- Full degradation level required
- Checkpoint approval required
- Domain reviewer sign-off (if configured)

## Failure Modes
| Issue                    | Outcome | Action              |
|--------------------------|---------|---------------------|
| Objective not met        | FAILED  | Retry or escalate   |
| Scope violation          | FAILED  | Fix scope or expand |
| Check failure            | FAILED  | Fix and retry       |
| Missing evidence         | FAILED  | Add evidence        |
| Verifier finds defects   | FAILED  | Address findings    |
| Scope expansion needed   | BLOCKED | Request approval    |
| Tools unavailable        | Degrade | Apply degradation   |
