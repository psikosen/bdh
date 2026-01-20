# gt-sling.md (Graph Theory Sling)

## Purpose
Lightweight reasoning framework for dependency and impact analysis within beads.

## Concepts

### File Dependency Graph
- Nodes: files in repository
- Edges: import/require/include relationships
- Use for: impact analysis, scope validation

### Change Impact Analysis
Given a set of changed files, compute:
1. Direct dependents (files that import changed files)
2. Transitive dependents (recursive closure)
3. Test coverage (which tests cover changed files)

## Operations

### scope_check(allowed_files, changed_files)
```
for each file in changed_files:
  if file not in allowed_files:
    return SCOPE_VIOLATION
return OK
```

### impact_radius(changed_files, depth=1)
```
impacted = set()
frontier = changed_files
for i in range(depth):
  next_frontier = set()
  for file in frontier:
    next_frontier |= dependents(file)
  impacted |= next_frontier
  frontier = next_frontier
return impacted
```

### minimal_test_set(changed_files)
```
tests = set()
for file in changed_files:
  tests |= tests_covering(file)
return tests
```

## Usage in Beads
- Executor uses gt-sling to verify scope compliance
- Verifier uses gt-sling to check for missing dependencies
- BD uses gt-sling to compute risk_level from impact_radius

## Limitations
- Requires static analysis support (may not be available for all languages)
- Dynamic dependencies not captured
- Apply degradation rules when gt-sling unavailable
