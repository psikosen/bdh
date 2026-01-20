# change-classifier.md

## Purpose
Classify changes by type and domain to determine risk and required checks.

## Change Types
| Type        | Description                              | Base Risk |
|-------------|------------------------------------------|-----------|
| bugfix      | Fixes incorrect behavior                 | MEDIUM    |
| feature     | Adds new functionality                   | MEDIUM    |
| refactor    | Restructures without behavior change     | LOW       |
| dependency  | Updates dependencies                     | MEDIUM    |
| migration   | Data or schema migration                 | HIGH      |
| security    | Security-related changes                 | HIGH      |
| performance | Performance optimization                 | MEDIUM    |
| infra       | Infrastructure/deployment changes        | HIGH      |
| docs        | Documentation only                       | LOW       |
| test        | Test additions/modifications             | LOW       |
| style       | Formatting, whitespace, naming           | LOW       |

## Change Domains
| Domain      | Paths/Patterns                           |
|-------------|------------------------------------------|
| api         | api/, routes/, controllers/, handlers/   |
| auth        | auth/, login/, session/, token/          |
| data        | models/, schema/, migrations/, db/       |
| ui          | components/, views/, pages/, templates/  |
| core        | lib/, src/core/, utils/                  |
| config      | config/, settings/, *.config.*           |
| ci          | .github/, .gitlab-ci.yml, Jenkinsfile    |
| deps        | package.json, Cargo.toml, requirements.txt|

## Classification Algorithm
```
1. Analyze diff for file paths
2. Match paths to domains
3. Analyze commit message/PR description for type keywords
4. If ambiguous, prompt for classification
5. Return (change_type, change_domain, base_risk)
```

## Classification Output
```yaml
classification:
  change_type: "<type>"
  change_domains: [<domains>]
  base_risk: "LOW|MEDIUM|HIGH"
  confidence: "high|medium|low"
  signals:
    - "<what indicated this classification>"
```

## Ambiguity Handling
If classification confidence < medium:
1. Present likely options to user
2. Request confirmation
3. Use confirmed classification
