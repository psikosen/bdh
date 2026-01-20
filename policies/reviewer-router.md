# reviewer-router.md

## Purpose
Route HIGH risk changes to appropriate reviewers.

## Triggers
- risk_level = HIGH
- api_change.breaking = true
- security domain touched
- migration type changes

## Reviewer Categories

### Domain Reviewers
```yaml
domain_reviewers:
  auth:
    expertise: "authentication, authorization, session management"
    triggers: ["auth/", "login/", "session/", "token/", "security/"]
  data:
    expertise: "database, migrations, data integrity"
    triggers: ["models/", "schema/", "migrations/", "db/"]
  api:
    expertise: "API design, contracts, versioning"
    triggers: ["api/", "routes/", "controllers/", "openapi*"]
  infra:
    expertise: "deployment, CI/CD, infrastructure"
    triggers: [".github/", "docker*", "k8s/", "terraform/"]
```

### Fallback
If no domain match: require human checkpoint approval

## Routing Algorithm
```
1. Identify change domains from classification
2. For each domain:
   a. Check if domain_reviewer configured
   b. If yes, add to required_reviewers
3. If required_reviewers empty:
   a. Fallback to human checkpoint
4. Request review from all required_reviewers
```

## Review Request Format
```yaml
review_request:
  bead_id: "<id>"
  risk_level: "HIGH"
  change_summary: "<brief>"
  domains: [<affected domains>]
  required_reviewers:
    - role: "<domain>"
      reason: "<why needed>"
  evidence_summary:
    files_changed: <count>
    breaking_changes: true|false
    security_implications: "<if any>"
```

## Review Outcomes
| Outcome  | Action                        |
|----------|-------------------------------|
| Approved | Proceed with bead             |
| Changes  | Apply feedback, re-verify     |
| Rejected | Mark BLOCKED, report reason   |

## Human Checkpoint Fallback
When no domain reviewer available:
```yaml
escalation:
  type: risk_threshold
  risk_level: "HIGH"
  message: "HIGH risk change requires human approval"
  summary: "<change summary>"
  options:
    - "Approve and proceed"
    - "Request more information"
    - "Reject change"
```
