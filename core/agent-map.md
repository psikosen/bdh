# agent-map.md

## Agent Roles

### BD (Bead Dispatcher)
- Role: Orchestrates bead execution
- Creates bead instances with proper scope
- Routes to executor and verifier
- Handles escalation and checkpoints
- See: core/bd.md

### Executor (subagent_bead_executor)
- Role: Implements bead objectives
- Respects allowed_files strictly
- Produces evidence artifacts
- See: agents/subagent_bead_executor.md

### Verifier (subagent_adversarial_verifier)
- Role: Adversarial review of executor work
- Read-only verification
- Validates scope, checks, evidence
- See: agents/subagent_adversarial_verifier.md

## Interaction Flow
```
User Request
    ↓
   BD (dispatch)
    ↓
Executor (implement)
    ↓
Verifier (validate)
    ↓
BD (outcome routing)
    ↓
PASSED | FAILED | BLOCKED
```

## Agent Permissions Matrix
| Agent    | Read  | Write | Bash  | Network | State |
|----------|-------|-------|-------|---------|-------|
| BD       | all   | meta  | no    | no      | yes   |
| Executor | scope | scope | check | no      | no    |
| Verifier | scope | no    | ro    | no      | no    |

## Communication Protocol
- BD → Executor: bead instance payload
- Executor → BD: evidence + outcomes
- BD → Verifier: evidence + bead instance
- Verifier → BD: verdict payload
