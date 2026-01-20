# checks-and-versions.md

## RULES
- Prefer repo targets/scripts; never invent commands.
- Choose least-expensive meaningful checks.
- Record selected checks, TOOL_MISSING, and tool_versions in evidence.
- If all required checks unavailable → escalation.tooling_missing + STOP.
- Missing tools → apply core/degradation.md.

## DEFAULT CHECKS (try in order)
Universal (if present): `just test|lint|fmt`, `make test|lint|fmt`, `task test|lint|fmt`, repo docs.
Rust: `cargo test`, `cargo clippy --all-targets --all-features`, `cargo fmt --check`
JS/TS: `npm|pnpm|yarn test`, `... run lint`, `... run format` (prefer scripts)
Dart/Flutter: `dart|flutter test`, `dart|flutter analyze`, `dart format --output=none --set-exit-if-changed .`
C#: `dotnet test`, `dotnet format --verify-no-changes` (if used), `dotnet build` (repo)
C++: use repo-documented build/test; only if configured: `cmake --build <build_dir>`, `ctest`
Postgres: use repo migration tooling; never prod; optional psql syntax-check if configured
Python: `pytest`, `ruff check` or `flake8`, `black --check` or `ruff format --check`
Go: `go test ./...`, `golangci-lint run`, `gofmt -l .`

## TOOL VERSIONS (for tools used)
```yaml
tool_versions:
  - tool: "<tool>"
    version: "<string|TOOL_MISSING>"
```

## Check Discovery Protocol
1. Look for package.json scripts, Makefile targets, justfile recipes, taskfile
2. Look for CI config (.github/workflows, .gitlab-ci.yml, etc.)
3. Look for tool configs (pyproject.toml, Cargo.toml, etc.)
4. Fall back to language defaults

## Check Selection Principles
- Prefer repo-defined over language defaults
- Prefer fast checks over slow (lint before test)
- Prefer checks with existing config
- Skip checks that duplicate functionality
