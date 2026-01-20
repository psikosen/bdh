# cache-keys.md

## Purpose
Ensure cache invalidation correctness when caching is used.

## Triggers
- Caching logic added or modified
- Cache key generation changes
- TTL changes
- Cache layer configuration

## Cache Key Principles
1. Keys must include ALL inputs that affect output
2. Keys must be deterministic
3. Keys should be human-readable for debugging
4. Keys should have bounded length

## Common Cache Key Patterns
```yaml
cache_patterns:
  user_data:
    key: "user:{user_id}:v{schema_version}"
    invalidate_on: ["user update", "schema migration"]

  computed_result:
    key: "compute:{input_hash}:v{algorithm_version}"
    invalidate_on: ["algorithm change"]

  api_response:
    key: "api:{endpoint}:{params_hash}:v{api_version}"
    invalidate_on: ["data change", "api version bump"]
```

## Cache Evidence
```yaml
cache_change:
  caches_affected:
    - name: "<cache name>"
      key_format: "<format>"
      previous_format: "<if changed>"
      ttl: "<duration>"
      invalidation_triggers: [<events>]
  invalidation_strategy: "<how stale data is handled>"
  migration_plan: "<if key format changed>"
```

## Cache Key Checklist
- [ ] All varying inputs included in key
- [ ] Version number in key (for format changes)
- [ ] TTL appropriate for data freshness needs
- [ ] Invalidation triggers documented
- [ ] Key collision risk assessed

## Anti-Patterns
- Timestamp in key (defeats caching)
- Missing version (breaks on format change)
- Overly broad keys (cache misses)
- Overly narrow keys (memory bloat)
- No invalidation strategy (stale data)

## Migration on Key Change
If cache key format changes:
1. Deploy with dual-read (old + new key)
2. Write to new key only
3. Wait for old TTL to expire
4. Remove old key logic
