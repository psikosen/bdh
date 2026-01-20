# testing.md

## Purpose
Testing guidelines and requirements.

## Test Categories

### Unit Tests
- Test individual functions/methods
- Mock external dependencies
- Fast execution
- High coverage of logic paths

### Integration Tests
- Test component interactions
- Use real dependencies where practical
- May be slower
- Focus on interfaces

### End-to-End Tests
- Test full workflows
- Use realistic scenarios
- Slowest category
- Focus on critical paths

## Test Requirements by Risk

### LOW Risk
- Existing tests must pass
- New code: tests encouraged but optional

### MEDIUM Risk
- Existing tests must pass
- New code: tests required for new functionality
- No test coverage regression

### HIGH Risk
- All tests must pass
- Comprehensive test coverage required
- Edge cases must be tested
- Error paths must be tested

## Test Evidence
```yaml
test_results:
  runner: "<test framework>"
  total: <int>
  passed: <int>
  failed: <int>
  skipped: <int>
  duration_ms: <int>
  coverage:
    lines: "<percentage|unknown>"
    branches: "<percentage|unknown>"
```

## Flaky Test Handling
- Identify and document flaky tests
- Don't retry to hide flakiness
- Fix or quarantine flaky tests
- Log flaky patterns for investigation

## Test Writing Guidelines
```yaml
test_structure:
  arrange: "Set up test data and conditions"
  act: "Execute the code under test"
  assert: "Verify the results"
  cleanup: "Restore state if needed"
```

## Coverage Expectations
- New code: aim for >80% line coverage
- Critical paths: aim for >90%
- Don't chase 100% (diminishing returns)
- Focus on meaningful coverage
