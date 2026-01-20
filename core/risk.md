# risk.md

## Risk Levels

### LOW
- Documentation-only changes
- Typo fixes
- Comment updates
- Test-only changes (adding tests, not modifying)
- Config changes with no runtime impact

### MEDIUM
- Bug fixes with clear scope
- Feature additions in isolated modules
- Refactoring within single file
- Dependency updates (minor/patch versions)
- Test modifications

### HIGH
- Security-related changes
- API/contract changes
- Database schema changes
- Authentication/authorization changes
- Dependency updates (major versions)
- Changes affecting multiple subsystems
- Performance-critical paths
- Data migration

## Risk Assessment Inputs
- change_type (from change-classifier)
- files_changed (count and criticality)
- impact_radius (from gt-sling)
- tests_affected (count)
- api_surface_touched (boolean)
- deps_touched (boolean)

## Risk Calculation
```yaml
risk_factors:
  base: <from change_type>
  modifiers:
    - impact_radius > 5: +1
    - api_surface_touched: +1
    - deps_major_update: +1
    - security_path: +2
    - no_test_coverage: +1
```

Sum modifiers:
- 0: keep base
- 1-2: raise one level
- 3+: HIGH

## Risk-Based Requirements
| Risk   | Checks      | Degradation | Approval      |
|--------|-------------|-------------|---------------|
| LOW    | lint+fmt    | minimal OK  | auto          |
| MEDIUM | +tests      | partial min | auto          |
| HIGH   | all         | full only   | checkpoint    |

## Escalation
- HIGH risk triggers policies/reviewer-router.md
- HIGH risk cannot pass under minimal degradation
