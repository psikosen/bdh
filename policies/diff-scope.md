# diff-scope.md

## Purpose
Track and validate that all changes stay within allowed scope.

## Diff Summary Structure
```yaml
diff_summary:
  files_changed:
    - path: "<file>"
      change_type: "added|modified|deleted|renamed"
      lines_added: <int>
      lines_removed: <int>
  total_files: <int>
  total_lines_added: <int>
  total_lines_removed: <int>
  scope_compliant: true|false
  out_of_scope_files: []  # populated if scope_compliant=false
```

## Scope Validation
```
for each file in files_changed:
  if file.path not in allowed_files:
    if file.path not matches any allowed glob:
      scope_compliant = false
      out_of_scope_files.append(file.path)
```

## Allowed Files Specification
allowed_files can be:
- Explicit paths: `["src/main.py", "tests/test_main.py"]`
- Glob patterns: `["src/**/*.py", "tests/**"]`
- Mixed: `["src/main.py", "tests/**/*.py"]`

## Scope Checking Rules
1. Check BEFORE modifying any file
2. If scope violation would occur, STOP
3. Emit scope_request for needed files
4. Wait for approval before proceeding

## Evidence Requirements
- diff_summary MUST match actual git diff
- Verifier will independently compute diff and compare
- Mismatch = automatic FAIL

## Diff Generation
```bash
# Generate diff for verification
git diff --stat HEAD~1..HEAD
git diff --numstat HEAD~1..HEAD
```

## Large Diff Handling
If total_lines_added + total_lines_removed > 500:
- Flag for extra scrutiny
- Consider breaking into smaller beads
- May trigger checkpoint for review
