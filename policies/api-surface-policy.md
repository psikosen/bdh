# api-surface-policy.md

## Purpose
Govern changes to API surfaces and contracts.

## Triggers
- Paths: api/, routes/, controllers/, handlers/, proto/
- Files: openapi*, swagger*, *.proto, *.graphql
- Keywords: endpoint, route, api, contract, schema, protobuf, openapi, swagger, response, request

## Required Evidence
```yaml
api_change:
  breaking: true|false
  endpoints_added:
    - path: "<endpoint>"
      method: "<HTTP method>"
      purpose: "<description>"
  endpoints_removed:
    - path: "<endpoint>"
      method: "<HTTP method>"
      migration_path: "<how clients migrate>"
  endpoints_modified:
    - path: "<endpoint>"
      changes: "<what changed>"
      breaking: true|false
  schema_changes:
    - type: "<entity>"
      change: "field_added|field_removed|field_modified|type_changed"
      breaking: true|false
  backward_compatible: true|false
  versioning_strategy: "<how versioned>"
```

## Breaking Change Definition
A change is BREAKING if it:
- Removes an endpoint
- Removes a required field from response
- Adds a required field to request
- Changes field type incompatibly
- Changes authentication/authorization requirements
- Changes error codes/formats

## Non-Breaking Changes
- Adding optional request fields
- Adding response fields
- Adding new endpoints
- Adding new error codes (if clients handle unknown codes)

## Breaking Change Protocol
1. Mark breaking: true in evidence
2. Risk automatically becomes HIGH
3. Checkpoint required
4. Document migration path
5. Consider versioning strategy

## Versioning Strategies
- URL versioning: /api/v1/, /api/v2/
- Header versioning: Accept-Version: v1
- Query parameter: ?version=1
- Deprecation period with sunset header

## API Documentation
- Update OpenAPI/Swagger spec if present
- Update API documentation
- Add changelog entry for breaking changes
