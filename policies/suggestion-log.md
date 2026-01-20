# suggestion-log.md

## Purpose
Track suggestions, improvements, and changes to the beads system.

## Log Format
```yaml
suggestions:
  - id: "<YYYY-MM-DD-NNN>"
    date: "<date>"
    category: "enhancement|bugfix|clarification|new-policy"
    status: "proposed|accepted|rejected|implemented"
    summary: "<brief description>"
    rationale: "<why suggested>"
    files_affected: [<paths>]
    discussion: "<notes>"
```

## Current Suggestions

### Implemented
(None yet - system newly created)

### Accepted
(None yet)

### Proposed
(None yet)

### Rejected
(None yet)

## Suggestion Process
1. Document suggestion in this log with status=proposed
2. Assess impact and backward compatibility
3. If accepted, implement and update status
4. If rejected, document reason

## Categories

### Enhancement
Improvements to existing functionality
- New evidence types
- Better validation
- Performance improvements

### Bugfix
Corrections to system behavior
- Logic errors in policies
- Missing edge cases
- Incorrect triggers

### Clarification
Documentation improvements
- Unclear language
- Missing examples
- Ambiguous rules

### New-Policy
Addition of new policies
- New change types
- New domains
- New checks

## Review Cadence
- Review suggestion-log.md periodically
- Archive implemented suggestions after one cycle
- Track patterns in rejected suggestions
