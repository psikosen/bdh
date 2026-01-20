# escalation.md

## Escalation Types

### checkpoint_required
User input needed to proceed.
```yaml
escalation:
  type: checkpoint_required
  reason: "<why>"
  options:
    - label: "<option 1>"
      action: "<what happens>"
    - label: "<option 2>"
      action: "<what happens>"
  context: "<relevant info>"
```

### scope_expansion
Need to modify files outside allowed_files.
```yaml
escalation:
  type: scope_expansion
  bead_id: "<id>"
  requested_files:
    - path: "<file>"
      reason: "<why needed>"
  risk_assessment: "LOW|MEDIUM|HIGH"
```

### tooling_missing
Required tools not available.
```yaml
escalation:
  type: tooling_missing
  tools:
    - name: "<tool>"
      purpose: "<what it does>"
  degradation_level: "partial|minimal"
  recommendation: "<suggested action>"
```

### ambiguity
Multiple valid interpretations exist.
```yaml
escalation:
  type: ambiguity
  question: "<what's unclear>"
  options:
    - interpretation: "<option A>"
      implications: "<what this means>"
    - interpretation: "<option B>"
      implications: "<what this means>"
```

### risk_threshold
Risk level requires approval.
```yaml
escalation:
  type: risk_threshold
  risk_level: "HIGH"
  factors:
    - "<factor 1>"
    - "<factor 2>"
  requires: "human_approval|domain_reviewer"
```

## Escalation Protocol
1. STOP current work immediately
2. Construct appropriate escalation payload
3. Present to user with clear options
4. WAIT for response (do not proceed)
5. Resume based on response

## Response Handling
| Response       | Action                    |
|----------------|---------------------------|
| Approved       | Proceed with request      |
| Denied         | Mark BLOCKED, report      |
| Modified       | Apply modifications, proceed |
| Deferred       | Park bead, continue others |

## Escalation Limits
- Max 3 escalations per bead before mandatory review
- Repeated same-type escalations trigger meta-checkpoint
