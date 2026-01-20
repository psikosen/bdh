# startup-index.md

## CORE (always read)
1. core/startup-index.md
2. core/agent-map.md
3. core/bd.md
4. core/gt-sling.md
5. core/bead-contract.md
6. core/risk.md
7. core/evidence.md
8. core/dod.md
9. core/degradation.md
10. core/escalation.md
11. core/enforcement.md

## CONDITIONAL (read only if trigger matches)
- overlays: repo signals → repo-conventions.md, language-index.md + languages/*
- code changes: non-doc OR change_type∈{bugfix,feature,refactor,dependency,migration,security,performance,infra} → best-practices.md, testing.md
- missing classification → policies/change-classifier.md
- any checks/run discovery → policies/checks-and-versions.md
- any file changes → policies/diff-scope.md
- deps touched → policies/dep-change-policy.md
- API/contract touched → policies/api-surface-policy.md
  triggers: paths api/routes/controllers/handlers/proto, openapi*, swagger*, *.proto OR keywords endpoint/route/api/contract/schema/protobuf/openapi/swagger/response/request
- risk_level=high → policies/reviewer-router.md (+ domain reviewers)
- autonomy enabled OR checkpoint trigger fires → policies/checkpoint.md
- after failure → policies/retry.md
- caching used → policies/cache-keys.md
- project override exists → .claude/project.md
- maintaining beads system → policies/process-policy.md, policies/suggestion-log.md

## GUARDRAIL
- max conditional reads per bead: 6
- if >6 needed → escalation.checkpoint_required and STOP
