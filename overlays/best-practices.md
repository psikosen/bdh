# best-practices.md

## Purpose
General best practices for code changes.

## Code Quality

### Minimal Changes
- Change only what's necessary
- Don't refactor unrelated code
- Keep PRs focused and reviewable

### Error Handling
- Return errors early
- Use appropriate error types
- Don't swallow exceptions silently
- Log errors with context

### Security
- Never hardcode secrets
- Validate all inputs at boundaries
- Use parameterized queries
- Escape output appropriately
- Follow OWASP guidelines

### Performance
- Measure before optimizing
- Consider memory and CPU
- Avoid premature optimization
- Document performance-critical code

## Code Style

### Naming
- Descriptive names over comments
- Consistent naming conventions
- Avoid abbreviations (except common ones)

### Structure
- Small, focused functions
- Single responsibility principle
- Appropriate abstraction level
- Minimal nesting depth

### Comments
- Explain why, not what
- Keep comments up to date
- Remove commented-out code
- Use TODO format: `TODO(author): description`

## Testing

### Coverage
- Test public interfaces
- Test edge cases
- Test error paths
- Don't test implementation details

### Quality
- Tests should be deterministic
- Tests should be independent
- Tests should be fast
- Tests should be readable

## Documentation

### Code
- Document public APIs
- Document complex algorithms
- Document non-obvious decisions

### Commits
- Clear, concise messages
- Reference issues/tickets
- Explain reasoning for changes
