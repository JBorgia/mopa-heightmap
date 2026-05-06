# Regenerating the TypeScript API types

The Angular SPA at `apps/web/` consumes a generated TypeScript definition file
(`apps/web/src/app/core/api/generated/api.d.ts`) derived from the FastAPI
schema. When [`apps/api/schemas.py`](../apps/api/schemas.py) changes — new
fields, renamed enums, removed routes — the generated file goes stale and
the CI drift guard ([`apps/web/scripts/check-openapi-drift.mjs`](../apps/web/scripts/check-openapi-drift.mjs))
fails the next frontend build.

## When to regenerate

Any time you touch:

* a Pydantic model in `apps/api/schemas.py`
* a `DEFAULT_SETTINGS` key in `mopa/service.py` that's mirrored
  into `HeightmapSettings`
* a route signature in `apps/api/routes/*.py`

…you should regenerate the TS types in the same commit.

## How to regenerate

From the repo root, with the Python venv activated and Node ≥ 20 on PATH:

```powershell
# 1. Regenerate openapi.json from the FastAPI schemas
.\.venv\Scripts\python.exe -m apps.api.export_openapi

# 2. Regenerate api.d.ts from openapi.json
cd apps\web
npm run sync:api-types
cd ..\..
```

`sync:api-types` is a single npm script that chains
`generate:openapi` → `generate:api-types`, so a one-liner from `apps/web` is:

```powershell
cd apps\web; npm run sync:api-types
```

The two updated files (`openapi.json` at the repo root and
`apps/web/src/app/core/api/generated/api.d.ts`) should be committed
together with the schema change.

## CI guard

`apps/web/scripts/check-openapi-drift.mjs` runs in the frontend CI job and
fails the build if the committed `api.d.ts` doesn't match what
`sync:api-types` would emit. So forgetting to regenerate is caught
automatically — but doing it locally first keeps the CI run green.

## When Node isn't available

If you're working in an environment without Node (e.g., a Python-only
sandbox), the schema additions still take effect server-side — only the
typed-frontend wiring is blocked. Capture the schema change in Python,
add a TODO referencing this doc, and regenerate the TS once you're back
on a workstation with Node.
