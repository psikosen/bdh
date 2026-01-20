# dep-change-policy.md

## Purpose
Govern dependency additions, removals, and updates.

## Triggers
- package.json, package-lock.json changes
- Cargo.toml, Cargo.lock changes
- requirements.txt, pyproject.toml changes
- go.mod, go.sum changes
- Gemfile, Gemfile.lock changes
- Any **/deps/**, **/vendor/** changes

## Required Evidence
```yaml
dependency_change:
  added:
    - pkg: "<name>"
      version: "<version>"
      purpose: "<why needed>"
      license: "<license>"
  removed:
    - pkg: "<name>"
      reason: "<why removed>"
  updated:
    - pkg: "<name>"
      from: "<old version>"
      to: "<new version>"
      breaking: true|false
      changelog_reviewed: true|false
  license_check: "pass|fail|skip"
  security_advisory_check: "pass|fail|skip"
```

## Version Update Rules
| Update Type | Risk    | Requirements              |
|-------------|---------|---------------------------|
| Patch       | LOW     | Run tests                 |
| Minor       | MEDIUM  | Run tests, review changes |
| Major       | HIGH    | Full review, checkpoint   |

## New Dependency Rules
- Document purpose (why existing deps don't suffice)
- Check license compatibility
- Check security advisories
- Prefer well-maintained packages (recent commits, active issues)
- Prefer smaller dependency trees

## Security Checks
```bash
# npm
npm audit

# Python
pip-audit

# Rust
cargo audit

# Go
govulncheck ./...
```

## License Compatibility
Allowed licenses (default):
- MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, MPL-2.0

Requires review:
- LGPL-*, GPL-* (copyleft implications)
- AGPL-* (network copyleft)
- Proprietary

## Lockfile Handling
- Always commit lockfile changes with dependency updates
- Review lockfile diff for unexpected transitive updates
- Flag if transitive deps have major version jumps
