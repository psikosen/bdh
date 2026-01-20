# Claude Beads System (Langpack v16)

## Prime Directive
Execute work via beads. Use subagents. Verify with adversarial review. Be concise.

## Startup (Mandatory)
1) Read `core/startup-index.md`
2) Read the mandatory core list from that index
3) Read only conditionals whose triggers match
4) Produce `startup_read_log` evidence (core/enforcement.md)

## Operating Rules
- Never expand scope beyond `allowed_files`. Use scope expansion protocol if needed.
- Prefer smallest safe change. Return errors early. Keep files small.
- Run required checks when possible; respect degradation rules if tooling is missing.
- Always run adversarial verifier after executor work.
- Evidence is mandatory; missing evidence is a fail.
