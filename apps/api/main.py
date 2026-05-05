"""FastAPI application entry point.

Launch with:
    uvicorn apps.api.main:app --host 127.0.0.1 --port 8000 --reload

Or serve the Angular build as static files in prod:
    uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
    (Angular dist/ is mounted at /)
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .routes import blob, export, mask, plan, profile, render, session, upload
from .schemas import ApiError, ErrorResponse

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MOPA Heightmap Studio API",
    version="9.0.0",
    description=(
        "Headless service layer powering the Angular + PrimeNG SPA. "
        "All depth inference and heightmap math lives in zoedepth.laser.*; "
        "this API is a thin HTTP adapter."
    ),
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# ---------------------------------------------------------------------------
# CORS — allow Angular dev server only in dev mode
# ---------------------------------------------------------------------------

_DEV_ORIGINS = ["http://localhost:4200", "http://127.0.0.1:4200"]
_IS_DEV = os.environ.get("MOPA_ENV", "dev") == "dev"

app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEV_ORIGINS if _IS_DEV else [],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(blob.router)
app.include_router(export.router)
app.include_router(mask.router)
app.include_router(plan.router)
app.include_router(profile.router)
app.include_router(render.router)
app.include_router(session.router)
app.include_router(upload.router)

# ---------------------------------------------------------------------------
# Global error handler — always return {error: {code, message, hint?}}
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def _global_error(request: Request, exc: Exception) -> JSONResponse:  # noqa: ARG001
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=ApiError(
                code="internal_error",
                message=str(exc),
                hint="Check server logs for the full traceback.",
            )
        ).model_dump(),
    )

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok"}

# ---------------------------------------------------------------------------
# Serve Angular SPA in production (mount last so API routes take priority)
# ---------------------------------------------------------------------------

_WEB_DIST = Path(__file__).parent.parent / "web" / "dist" / "web" / "browser"
if _WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=str(_WEB_DIST), html=True), name="spa")
