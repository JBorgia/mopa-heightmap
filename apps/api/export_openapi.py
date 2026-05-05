"""Generate openapi.json from the FastAPI app and write to apps/web/openapi.json.

Used by the CI drift-guard:
  python -m apps.api.export_openapi
  npx openapi-typescript apps/web/openapi.json -o apps/web/src/app/core/api/generated/api.d.ts
  git diff --exit-code apps/web/src/app/core/api/generated/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .main import app

_OUT = Path(__file__).parent.parent / "web" / "openapi.json"


def main() -> None:
    schema = app.openapi()
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"Written: {_OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
