# bead-contract.md

## Bead Instance (Input)
A bead is a bounded unit of work with explicit scope and checks.

Required fields:
- bead.id
- bead.objective
- allowed_files
- expected_outputs
- stop_conditions
- required_checks (or empty, in which case checks-and-versions defaults apply)
- risk_level (low|medium|high) OR enough info for change-classifier

Optional fields:
- change_domain / change_type
- invariants
- evidence_requirements
- custom_dod (additive only)

## Scope Rules
- Only modify files in allowed_files.
- If you discover out-of-scope files are required:
  1) STOP immediately
  2) Do not modify out-of-scope files
  3) Emit scope expansion request:

```yaml
scope_request:
  bead_id: "<id>"
  reason: "<why required>"
  requested_files:
    - "<path>"
  justification: "<short>"
  risk_assessment: "LOW|MEDIUM|HIGH"
```

If denied: mark bead BLOCKED (not FAILED).

## Outcomes
- PASSED: verifier confirms evidence + checks + scope
- FAILED: verifier finds defects or missing evidence (retriable)
- BLOCKED: requires scope expansion or external dependency (not retriable without intervention)
