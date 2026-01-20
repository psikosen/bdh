# process-policy.md

## Purpose
Govern changes to the beads system itself.

## Triggers
- Any file in core/, agents/, policies/, overlays/ modified
- Maintaining beads system

## Meta-Change Rules
Changes to the beads system require:
1. Clear rationale documented
2. Backward compatibility assessment
3. Impact on existing beads
4. Update to suggestion-log.md

## Change Categories

### Core Changes
Files: core/*.md
- Highest scrutiny
- Must not break existing bead contracts
- Require checkpoint approval
- Document in suggestion-log.md

### Agent Changes
Files: agents/*.md
- Must maintain role boundaries
- Test against sample beads if possible
- Document capability changes

### Policy Changes
Files: policies/*.md
- Assess impact on existing workflows
- Update triggers in startup-index.md if needed
- Document in suggestion-log.md

### Overlay Changes
Files: overlays/*.md
- Lower risk (repo-specific)
- Still require evidence

## Evidence for Process Changes
```yaml
process_change:
  files_modified: [<paths>]
  rationale: "<why change needed>"
  backward_compatible: true|false
  breaking_changes:
    - change: "<what breaks>"
      migration: "<how to adapt>"
  impact_assessment: "<who/what affected>"
```

## Version Compatibility
- Major: Breaking changes to bead contract
- Minor: New optional fields, new policies
- Patch: Clarifications, typo fixes

## Testing Process Changes
Before finalizing:
1. Dry-run against sample bead
2. Verify startup-index triggers
3. Check enforcement rules
4. Validate evidence requirements
