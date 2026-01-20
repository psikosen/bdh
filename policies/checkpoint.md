# checkpoint.md

## TRIGGERS
- autonomy: 5 consecutive beads since last checkpoint
- autonomy: 3+ MEDIUM beads since last checkpoint
- any bead BLOCKED
- degradation MINIMAL
- HIGH risk requires human approval (instead of domain reviewer)
- bounded ambiguity resolvable via 2-5 options
- startup requires >6 conditional reads

## ASK RULES
1 question; 2-5 options; actionable; STOP after payload.

## PAYLOAD
Use escalation.type=checkpoint_required (core/escalation.md).

```yaml
checkpoint:
  trigger: "<which trigger fired>"
  question: "<single clear question>"
  options:
    - label: "<option>"
      description: "<what happens if selected>"
  context:
    beads_completed: <int>
    current_degradation: "<level>"
    pending_issues: [<list>]
```

## Checkpoint Protocol
1. Detect trigger condition
2. STOP current execution
3. Construct checkpoint payload
4. Present to user
5. WAIT for response
6. Resume based on selection

## Response Handling
| Selection           | Action                          |
|---------------------|---------------------------------|
| Continue            | Resume execution                |
| Modify scope        | Update allowed_files, continue  |
| Abort               | Mark remaining beads CANCELLED  |
| Switch approach     | Reconstruct beads, restart      |

## Checkpoint State
```yaml
checkpoint_state:
  last_checkpoint: "<timestamp>"
  beads_since: <int>
  medium_beads_since: <int>
  blocked_beads: [<ids>]
```

## Anti-Patterns
- Do NOT checkpoint for trivial decisions
- Do NOT checkpoint repeatedly for same issue
- Do NOT proceed after checkpoint without response
- Do NOT bundle multiple questions in one checkpoint
