"""ARQ job-dispatch helper for the FastAPI request handlers.

Usage:
    from .queue import enqueue_sculptok, get_job_result

    job_id = await enqueue_sculptok(image_id="abc123", ...)
    result = await get_job_result(job_id)  # None while pending

When Redis is not configured (REDIS_URL absent or connection fails), both
functions return None so callers can fall back to synchronous execution.
"""
from __future__ import annotations

import os
from typing import Any, Optional

_REDIS_URL = os.environ.get("REDIS_URL", "")

_pool = None
_pool_failed = False


async def _get_pool():
    global _pool, _pool_failed
    if _pool_failed or not _REDIS_URL:
        return None
    if _pool is not None:
        return _pool
    try:
        from arq import create_pool  # type: ignore[import]
        from arq.connections import RedisSettings  # type: ignore[import]
        _pool = await create_pool(RedisSettings.from_dsn(_REDIS_URL))
        return _pool
    except Exception:
        _pool_failed = True
        return None


async def enqueue_sculptok(**kwargs: Any) -> Optional[str]:
    """Enqueue a sculptok_generate_task. Returns job_id or None if Redis unavailable."""
    pool = await _get_pool()
    if pool is None:
        return None
    from .worker import sculptok_generate_task
    job = await pool.enqueue_job("sculptok_generate_task", **kwargs)
    return job.job_id if job else None


async def get_job_result(job_id: str) -> Optional[dict]:
    """Poll a job by id. Returns None while pending, result dict when done."""
    pool = await _get_pool()
    if pool is None:
        return None
    try:
        from arq.jobs import Job, JobStatus  # type: ignore[import]
        job = Job(job_id, pool)
        status = await job.status()
        if status in (JobStatus.complete,):
            return await job.result()
        if status in (JobStatus.not_found,):
            return {"error": "Job not found or expired."}
        return None  # still running
    except Exception:
        return None
