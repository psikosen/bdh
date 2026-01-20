# retry.md

## Purpose
Govern retry behavior after bead failures.

## Triggers
- Bead outcome = FAILED
- Transient errors (network, timeout)
- Check failures

## Retry Policy
```yaml
retry_policy:
  max_attempts: 3
  backoff: "exponential"
  base_delay_ms: 1000
  max_delay_ms: 30000
  retryable_failures:
    - "transient_error"
    - "timeout"
    - "check_flaky"
  non_retryable_failures:
    - "scope_violation"
    - "missing_evidence"
    - "security_violation"
```

## Retry Decision Tree
```
if failure_type in non_retryable_failures:
  return NO_RETRY, escalate
elif attempts >= max_attempts:
  return NO_RETRY, escalate
elif failure_type == "check_failure":
  return RETRY_WITH_FIX
elif failure_type == "transient_error":
  return RETRY_WITH_BACKOFF
else:
  return NO_RETRY, escalate
```

## Failure Categories

### Retryable
- Network timeouts
- Flaky test failures (if pattern detected)
- Resource contention
- Rate limiting

### Non-Retryable
- Scope violations
- Missing evidence
- Logic errors
- Security findings
- Verifier high-severity findings

## Retry Evidence
```yaml
retry_record:
  attempt: <int>
  failure_type: "<type>"
  failure_details: "<what failed>"
  action_taken: "<what changed before retry>"
  outcome: "success|failure"
```

## Escalation After Retries
If max_attempts reached:
```yaml
escalation:
  type: checkpoint_required
  reason: "Bead failed after max retries"
  attempts: <int>
  failures:
    - attempt: 1
      failure: "<summary>"
    - attempt: 2
      failure: "<summary>"
  options:
    - "Provide guidance and retry"
    - "Modify scope and retry"
    - "Abort bead"
```

## Flaky Detection
If same check fails intermittently across retries:
- Flag as potentially flaky
- Log pattern for future reference
- Consider skipping with FLAKY_SKIP (requires checkpoint approval)
