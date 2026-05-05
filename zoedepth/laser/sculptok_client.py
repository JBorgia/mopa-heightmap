"""Sculptok API client — automated bring-your-own-relief generation.

Sculptok publishes a credit-based REST API at
``https://api.sculptok.com/api-open`` that produces high-quality bas-relief
depth maps from input images. The authoritative source for the relief in
our pivot architecture: send a photo to Sculptok, poll for completion,
download the resulting heightmap PNG, then feed it straight into
:mod:`zoedepth.laser.external_heightmap` and the rest of the engraving
toolchain. End-to-end automated:

    photo  ->  Sculptok API  ->  heightmap PNG  ->  external_heightmap mode
                                                ->  pass-stack + .lbrn2

Authentication: header ``apikey: <user_api_key>``. The key is provisioned
through https://www.sculptok.com/api on a paid plan (PRO/$19/mo and up
recommended for any sustained use; ULTIMATE/$75/mo for "unlimited" jobs).

Sculptok's pricing page warns that *high-frequency API calls* will get
your account suspended. This client therefore:

    * sets an honest ``User-Agent`` string identifying mopa-heightmap,
    * rate-limits in the polling loop (``DEFAULT_POLL_INTERVAL_S = 3.0``),
    * surfaces credit-balance checks to the caller before submission,
    * never auto-retries on transient errors.

API surface (May 2026 docs):

    POST /api-open/image/upload          (multipart file)        -> src URL
    POST /api-open/draw/prompt           (JSON imageUrl + opts)  -> promptId
    GET  /api-open/draw/prompt?uuid=...                           -> status + imgRecords
    GET  /api-open/point/info                                     -> credits
    GET  /api-open/point/page                                     -> credits history
    GET  /api-open/image/page                                     -> drawing history

Pricing (May 2026):
    style=normal/portrait/sketch     : 10 credits
    style=pro                        : 15 credits
    style=pro AND draw_hd=4k         : 30 credits
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


__all__ = [
    "SculptokClient",
    "SculptokAPIError",
    "SculptokInsufficientCreditsError",
    "SculptokDepthMapParams",
    "SculptokHDFixParams",
    "SculptokThreeDParams",
    "SculptokSTLParams",
    "SculptokTaskStatus",
    "BASE_URL",
    "DEFAULT_USER_AGENT",
    "DEFAULT_POLL_INTERVAL_S",
    "DEFAULT_POLL_TIMEOUT_S",
    "STATUS_QUEUED",
    "STATUS_PROCESSING",
    "STATUS_COMPLETED",
    "PROMPT_PRICE_NORMAL",
    "PROMPT_PRICE_PRO",
    "PROMPT_PRICE_PRO_4K",
    "PROMPT_PRICE_HD_FIX",
    "PROMPT_PRICE_3D",
    "PROMPT_PRICE_STL",
]


BASE_URL: str = "https://api.sculptok.com/api-open"

# Self-identifying UA so Sculptok's logs can attribute traffic to this
# integration if their team ever needs to reach us about rate-limit
# policy or abuse signals. Honest > clever.
DEFAULT_USER_AGENT: str = "mopa-heightmap/1.0 (+https://github.com/jonathanborgia/mopa-heightmap)"

# Polling cadence for long-running jobs. Sculptok's UI suggests jobs
# complete in seconds; we still poll slowly to be a good API citizen.
DEFAULT_POLL_INTERVAL_S: float = 3.0
DEFAULT_POLL_TIMEOUT_S: float = 300.0

# Status codes from /draw/prompt (GET). Inferred from the example payload
# in the docs (status: 2 with full imgRecords = completed). The other
# values aren't formally enumerated but follow the standard "queued <
# processing < done < failed" pattern.
STATUS_QUEUED: int = 0
STATUS_PROCESSING: int = 1
STATUS_COMPLETED: int = 2
# Anything > 2 we treat as failure.

# Prices in credits (May 2026 docs). Surfaced here so callers can do
# pre-flight balance checks without hardcoding the numbers.
PROMPT_PRICE_NORMAL: int = 10
PROMPT_PRICE_PRO: int = 15
PROMPT_PRICE_PRO_4K: int = 30
PROMPT_PRICE_HD_FIX: int = 2
PROMPT_PRICE_3D: int = 10
PROMPT_PRICE_STL: int = 3


class SculptokAPIError(RuntimeError):
    """Raised when Sculptok responds with ``code != 0`` or HTTP non-2xx."""

    def __init__(self, message: str, *, code: Optional[int] = None, raw: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.raw = raw


class SculptokInsufficientCreditsError(SculptokAPIError):
    """The account doesn't have enough credits to submit this job."""


@dataclass(frozen=True)
class SculptokDepthMapParams:
    """Parameters for ``POST /draw/prompt``.

    Defaults are tuned for engraving:

        * ``style="pro"`` — highest-quality model (15 credits).
        * ``ext_info="16bit"`` — single 16-bit PNG return (best precision
          for laser-engraving heightmaps; non-pro modes return 3 8-bit
          variants).
        * ``version="1.5"`` — newer pro-model release.
        * ``draw_hd="2k"`` — half the cost of 4k while still being plenty
          for laser engraving on most stock sizes.

    Use :meth:`expected_cost` to predict the credit hit before submission.
    """

    style: str = "pro"
    hd_fix: str = "manual"
    optimal_size: str = "true"
    ext_info: str = "16bit"
    version: str = "1.5"
    draw_hd: str = "2k"

    def __post_init__(self) -> None:
        if self.style not in {"normal", "portrait", "sketch", "pro"}:
            raise ValueError(f"style must be one of normal|portrait|sketch|pro; got {self.style!r}")
        if self.hd_fix not in {"auto", "manual"}:
            raise ValueError(f"hd_fix must be auto|manual; got {self.hd_fix!r}")
        if self.optimal_size not in {"true", "false"}:
            raise ValueError(f"optimal_size must be 'true'|'false'; got {self.optimal_size!r}")
        if self.ext_info not in {"false", "8bit", "16bit", "exr"}:
            raise ValueError(
                f"ext_info must be false|8bit|16bit|exr; got {self.ext_info!r}"
            )
        if self.ext_info == "exr" and self.style != "pro":
            raise ValueError("ext_info=exr is only available when style=pro")
        if self.version not in {"1.0", "1.5"}:
            raise ValueError(f"version must be 1.0|1.5; got {self.version!r}")
        if self.draw_hd not in {"2k", "4k"}:
            raise ValueError(f"draw_hd must be 2k|4k; got {self.draw_hd!r}")

    def to_request_body(self, image_url: str) -> Dict[str, Any]:
        body: Dict[str, Any] = {"imageUrl": image_url, "style": self.style}
        # Send the pro-only fields only when actually relevant. Some
        # APIs reject "version" / "draw_hd" when style != pro.
        if self.style == "pro":
            body["version"] = self.version
            body["draw_hd"] = self.draw_hd
        body["hd_fix"] = self.hd_fix
        body["optimal_size"] = self.optimal_size
        body["extInfo"] = self.ext_info
        return body

    def expected_cost(self) -> int:
        """Return credits the submission will cost based on Sculptok's pricing."""
        if self.style != "pro":
            return PROMPT_PRICE_NORMAL
        return PROMPT_PRICE_PRO_4K if self.draw_hd == "4k" else PROMPT_PRICE_PRO


@dataclass(frozen=True)
class SculptokHDFixParams:
    """Parameters for ``POST /draw/hd/prompt`` (background removal + HD restore).

    Both fields are optional; supply one or both:

        * ``hd_fix=True`` runs the HD restoration pass.
        * ``remove_back="general" | "anime"`` runs the bg-removal pass
          tuned for that subject class. ``None`` skips bg removal.

    Pricing: 2 credits per call regardless of which sub-tasks run.
    """

    hd_fix: bool = True
    remove_back: Optional[str] = None  # None | "general" | "anime"

    def __post_init__(self) -> None:
        if self.remove_back not in (None, "general", "anime"):
            raise ValueError(
                f"remove_back must be None|general|anime; got {self.remove_back!r}"
            )

    def to_request_body(self, image_url: str) -> Dict[str, Any]:
        body: Dict[str, Any] = {"imageUrl": image_url}
        body["hdFix"] = "true" if self.hd_fix else "false"
        if self.remove_back is not None:
            body["removeBack"] = self.remove_back
        return body

    def expected_cost(self) -> int:
        return PROMPT_PRICE_HD_FIX


@dataclass(frozen=True)
class SculptokThreeDParams:
    """Parameters for ``POST /draw/3d/prompt`` (full 3D mesh generation).

    ``hd_fix`` (oddly named — it's really the precision tier in the docs)
    drives output quality:

        * ``"basic"`` (default): fastest / cheapest geometry.
        * ``"standard"``: middle ground.
        * ``"high"``: most detail.

    Pricing: 10 credits per call. Note: per Sculptok's pricing page,
    "Advanced 3D Model in 3D Generation does not qualify for Credit
    exemption" even on ULTIMATE, so this always bills against the user's
    balance.
    """

    hd_fix: str = "basic"

    def __post_init__(self) -> None:
        if self.hd_fix not in {"basic", "standard", "high"}:
            raise ValueError(
                f"hd_fix must be basic|standard|high; got {self.hd_fix!r}"
            )

    def to_request_body(self, image_url: str) -> Dict[str, Any]:
        return {"imageUrl": image_url, "hd_fix": self.hd_fix}

    def expected_cost(self) -> int:
        return PROMPT_PRICE_3D


@dataclass(frozen=True)
class SculptokSTLParams:
    """Parameters for ``POST /draw/stl/prompt`` (image → STL mesh).

    Sculptok's image-to-STL service rasterises a depth map straight to
    a printable STL. Useful as an alternative to our local
    :func:`apps.api.routes.export.export_stl` route — particularly when
    the user already has a Sculptok-generated depth map and wants the
    STL alongside the heightmap PNG.

    Pricing: 3 credits per call.
    """

    width_mm: float = 120.0          # 40-240 mm
    min_thickness: float = 1.6       # 0.4-8 mm; brightest area
    max_thickness: float = 5.0       # 0.4-25 mm; darkest area
    invert: bool = False             # True = white deep, black shallow
    scale_image: float = 50.0        # 0-100 percent

    def __post_init__(self) -> None:
        if not 40.0 <= self.width_mm <= 240.0:
            raise ValueError(f"width_mm must be 40..240; got {self.width_mm}")
        if not 0.4 <= self.min_thickness <= 8.0:
            raise ValueError(
                f"min_thickness must be 0.4..8; got {self.min_thickness}"
            )
        if not 0.4 <= self.max_thickness <= 25.0:
            raise ValueError(
                f"max_thickness must be 0.4..25; got {self.max_thickness}"
            )
        if self.max_thickness <= self.min_thickness:
            raise ValueError(
                "max_thickness must exceed min_thickness "
                f"(got {self.min_thickness}/{self.max_thickness})"
            )
        if not 0.0 <= self.scale_image <= 100.0:
            raise ValueError(
                f"scale_image must be 0..100; got {self.scale_image}"
            )

    def to_request_body(self, image_url: str) -> Dict[str, Any]:
        return {
            "image_url": image_url,
            "width_mm": float(self.width_mm),
            "min_thickness": float(self.min_thickness),
            "max_thickness": float(self.max_thickness),
            "invert": bool(self.invert),
            "scale_image": float(self.scale_image),
        }

    def expected_cost(self) -> int:
        return PROMPT_PRICE_STL


@dataclass(frozen=True)
class SculptokTaskStatus:
    """Decoded ``GET /draw/prompt`` response."""

    prompt_id: str
    status: int                     # 0=queued, 1=processing, 2=completed
    queue_position: Optional[int]   # only meaningful when queued
    current_step: Optional[int]     # 0..3 (model's internal step)
    upload_image_url: Optional[str]
    image_results: List[str]        # imgRecords (length up to 3)
    raw: Dict[str, Any]             # raw payload for debugging

    @property
    def is_completed(self) -> bool:
        return self.status == STATUS_COMPLETED and bool(self.image_results)

    @property
    def is_failed(self) -> bool:
        return self.status > STATUS_COMPLETED


class SculptokClient:
    """Thin REST wrapper around the Sculptok API.

    Lazy-imports ``requests`` so the module is importable in environments
    that don't have it (the rest of mopa-heightmap doesn't need outbound
    HTTP). One instance per API key is the right granularity.
    """

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = BASE_URL,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout_s: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._user_agent = user_agent
        self._timeout_s = float(timeout_s)
        self._session: Any = None  # lazy

    # ----------------------------------------------------------------- session

    def _ensure_session(self) -> Any:
        """Lazy-create a ``requests.Session`` with our auth + UA headers."""
        if self._session is None:
            import requests

            self._session = requests.Session()
            self._session.headers.update({
                "apikey": self._api_key,
                "User-Agent": self._user_agent,
            })
        return self._session

    def _check_envelope(self, payload: Dict[str, Any]) -> Any:
        """Validate Sculptok's standard envelope and return ``data``.

        Envelope shape (from docs):

            { "code": 0, "msg": "success", "data": { ... } }

        ``code != 0`` → raise. Insufficient-credits errors are surfaced as
        a dedicated subclass so callers can handle them gracefully (e.g.
        prompt for upgrade) rather than treating them as generic errors.
        """
        code = int(payload.get("code", -1))
        msg = str(payload.get("msg", ""))
        if code == 0:
            return payload.get("data")
        if "credit" in msg.lower() or "point" in msg.lower() or code in (402,):
            raise SculptokInsufficientCreditsError(msg, code=code, raw=payload)
        raise SculptokAPIError(msg or f"sculptok error code={code}", code=code, raw=payload)

    # ------------------------------------------------------------- credits API

    def get_credits(self) -> int:
        """Return the current credit balance (``GET /point/info``)."""
        sess = self._ensure_session()
        r = sess.get(
            f"{self._base_url}/point/info",
            timeout=self._timeout_s,
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        data = self._check_envelope(r.json())
        return int(data.get("point", 0))

    def get_credits_history(self, *, limit: int = 5, page: int = 1) -> List[Dict[str, Any]]:
        """Return the credit-change history (``GET /point/page``)."""
        sess = self._ensure_session()
        r = sess.get(
            f"{self._base_url}/point/page",
            timeout=self._timeout_s,
            params={"limit": int(limit), "page": str(page)},
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        data = self._check_envelope(r.json()) or {}
        return list(data.get("list", []))

    # ------------------------------------------------------- upload + submit

    def upload_image(self, image_path: str | Path) -> str:
        """Upload a source image; return the URL Sculptok stored it at.

        Endpoint: ``POST /image/upload`` (multipart/form-data). The
        returned URL is the value to pass as ``imageUrl`` to
        :meth:`submit_depth_map`.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"upload source not found: {path}")
        sess = self._ensure_session()
        with path.open("rb") as fh:
            r = sess.post(
                f"{self._base_url}/image/upload",
                timeout=self._timeout_s,
                files={"file": (path.name, fh, "application/octet-stream")},
            )
        r.raise_for_status()
        data = self._check_envelope(r.json()) or {}
        src = data.get("src")
        if not src:
            raise SculptokAPIError(
                "upload succeeded but response had no 'src' URL", raw=data,
            )
        return str(src)

    def submit_depth_map(
        self,
        image_url: str,
        params: SculptokDepthMapParams = SculptokDepthMapParams(),
    ) -> str:
        """Submit a depth-map generation job; return the ``promptId`` task id.

        Endpoint: ``POST /draw/prompt``. The ``promptId`` is the handle
        used by :meth:`get_drawing_status` to poll for completion.
        """
        sess = self._ensure_session()
        r = sess.post(
            f"{self._base_url}/draw/prompt",
            timeout=self._timeout_s,
            json=params.to_request_body(image_url),
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        data = self._check_envelope(r.json()) or {}
        prompt_id = data.get("promptId")
        if not prompt_id:
            raise SculptokAPIError(
                "submit succeeded but response had no 'promptId'", raw=data,
            )
        return str(prompt_id)

    def get_drawing_status(self, prompt_id: str) -> SculptokTaskStatus:
        """Poll a submitted job (``GET /draw/prompt?uuid=...``)."""
        sess = self._ensure_session()
        r = sess.get(
            f"{self._base_url}/draw/prompt",
            timeout=self._timeout_s,
            params={"uuid": prompt_id},
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        data = self._check_envelope(r.json()) or {}
        return SculptokTaskStatus(
            prompt_id=str(data.get("promptId", prompt_id)),
            status=int(data.get("status", -1)),
            queue_position=(
                int(data["position"]) if "position" in data and data["position"] is not None
                else None
            ),
            current_step=(
                int(data["currentStep"]) if "currentStep" in data
                and data["currentStep"] is not None else None
            ),
            upload_image_url=data.get("upImageUrl"),
            image_results=list(data.get("imgRecords") or []),
            raw=data,
        )

    # ------------------------------------------- background removal / HD fix

    def submit_hd_fix(
        self,
        image_url: str,
        params: SculptokHDFixParams = SculptokHDFixParams(),
    ) -> str:
        """Submit a Background-Removal + HD-Restoration job.

        Endpoint: ``POST /draw/hd/prompt``. Returns the ``promptId`` to
        poll with :meth:`get_drawing_status`. Costs 2 credits regardless
        of which sub-tasks (HD fix / bg removal) run.
        """
        sess = self._ensure_session()
        r = sess.post(
            f"{self._base_url}/draw/hd/prompt",
            timeout=self._timeout_s,
            json=params.to_request_body(image_url),
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        data = self._check_envelope(r.json()) or {}
        prompt_id = data.get("promptId")
        if not prompt_id:
            raise SculptokAPIError(
                "hd-fix submit succeeded but response had no 'promptId'", raw=data,
            )
        return str(prompt_id)

    # ------------------------------------------------------ 3D mesh draw

    def submit_3d_draw(
        self,
        image_url: str,
        params: SculptokThreeDParams = SculptokThreeDParams(),
    ) -> str:
        """Submit a 3D-mesh generation job.

        Endpoint: ``POST /draw/3d/prompt``. Returns the ``promptId`` to
        poll with :meth:`get_drawing_status`. Costs 10 credits per call.
        Per Sculptok's pricing page, this *does not* qualify for the
        ULTIMATE plan's "unlimited" credit exemption.
        """
        sess = self._ensure_session()
        r = sess.post(
            f"{self._base_url}/draw/3d/prompt",
            timeout=self._timeout_s,
            json=params.to_request_body(image_url),
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        data = self._check_envelope(r.json()) or {}
        prompt_id = data.get("promptId")
        if not prompt_id:
            raise SculptokAPIError(
                "3D submit succeeded but response had no 'promptId'", raw=data,
            )
        return str(prompt_id)

    # ----------------------------------------------------- image-to-STL

    def submit_image_to_stl(
        self,
        image_url: str,
        params: SculptokSTLParams = SculptokSTLParams(),
    ) -> str:
        """Submit an Image-to-STL job (Sculptok's ready-to-print pipeline).

        Endpoint: ``POST /draw/stl/prompt``. Costs 3 credits per call.
        Returns the ``promptId`` to poll; on completion, the STL URL
        appears in :attr:`SculptokTaskStatus.image_results`.
        """
        sess = self._ensure_session()
        r = sess.post(
            f"{self._base_url}/draw/stl/prompt",
            timeout=self._timeout_s,
            json=params.to_request_body(image_url),
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        data = self._check_envelope(r.json()) or {}
        prompt_id = data.get("promptId")
        if not prompt_id:
            raise SculptokAPIError(
                "STL submit succeeded but response had no 'promptId'", raw=data,
            )
        return str(prompt_id)

    # ----------------------------------------------------- drawing history

    def get_drawing_history(
        self, *, limit: int = 5, page: int = 1,
    ) -> List[Dict[str, Any]]:
        """List previously generated images (``GET /image/page``).

        Returns the raw list of records (each has ``id``, ``userId``,
        ``imgUrl``, ``createDate``); callers use ``imgUrl`` to download
        a previous result without re-submitting (and re-paying).
        """
        sess = self._ensure_session()
        r = sess.get(
            f"{self._base_url}/image/page",
            timeout=self._timeout_s,
            params={"limit": int(limit), "page": int(page)},
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        data = self._check_envelope(r.json()) or {}
        return list(data.get("list", []))

    # ----------------------------------------------------- polling helper

    def wait_for_completion(
        self,
        prompt_id: str,
        *,
        interval_s: float = DEFAULT_POLL_INTERVAL_S,
        timeout_s: float = DEFAULT_POLL_TIMEOUT_S,
        on_status: Optional[Any] = None,
    ) -> SculptokTaskStatus:
        """Poll :meth:`get_drawing_status` until the job finishes.

        Returns the final :class:`SculptokTaskStatus`. Raises
        :class:`TimeoutError` after ``timeout_s`` seconds, or
        :class:`SculptokAPIError` if the task ends in failure.

        ``on_status`` is an optional callback invoked once per poll with
        the latest :class:`SculptokTaskStatus`; useful for surfacing
        progress in a CLI / wizard.
        """
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            status = self.get_drawing_status(prompt_id)
            if on_status is not None:
                try:
                    on_status(status)
                except Exception:
                    # Status callbacks are convenience; never let one
                    # take down the polling loop.
                    pass
            if status.is_completed:
                return status
            if status.is_failed:
                raise SculptokAPIError(
                    f"task {prompt_id!r} failed (status={status.status})",
                    code=status.status, raw=status.raw,
                )
            time.sleep(float(interval_s))
        raise TimeoutError(
            f"sculptok task {prompt_id!r} did not complete within {timeout_s}s"
        )

    # ----------------------------------------------------- end-to-end helper

    def generate_heightmap(
        self,
        photo_path: str | Path,
        *,
        params: SculptokDepthMapParams = SculptokDepthMapParams(),
        out_path: Optional[str | Path] = None,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        poll_timeout_s: float = DEFAULT_POLL_TIMEOUT_S,
        check_credits: bool = True,
        on_status: Optional[Any] = None,
    ) -> Path:
        """One-shot: upload → submit → poll → download. Returns local PNG path.

        Convenience wrapper. Saves to ``out_path`` if given, otherwise to
        ``<photo>_sculptok.png`` next to the source.

        When ``check_credits=True`` (the default) the client fetches the
        current balance first and raises
        :class:`SculptokInsufficientCreditsError` *before* uploading if
        the credit balance is below the expected cost — saves a roundtrip.
        """
        photo_path = Path(photo_path)
        if not photo_path.exists():
            raise FileNotFoundError(f"photo not found: {photo_path}")
        if out_path is None:
            out_path = photo_path.with_name(photo_path.stem + "_sculptok.png")
        out_path = Path(out_path)

        if check_credits:
            balance = self.get_credits()
            cost = params.expected_cost()
            if balance < cost:
                raise SculptokInsufficientCreditsError(
                    f"need {cost} credits for {params.style}/{params.draw_hd}, "
                    f"only {balance} available",
                    raw={"balance": balance, "cost": cost},
                )

        image_url = self.upload_image(photo_path)
        prompt_id = self.submit_depth_map(image_url, params=params)
        status = self.wait_for_completion(
            prompt_id,
            interval_s=poll_interval_s,
            timeout_s=poll_timeout_s,
            on_status=on_status,
        )
        if not status.image_results:
            raise SculptokAPIError(
                f"task {prompt_id!r} completed but had no image results",
                raw=status.raw,
            )

        # When ext_info is set (16bit/8bit/exr) only one image is returned;
        # otherwise three variants are produced and we take the first.
        result_url = status.image_results[0]
        sess = self._ensure_session()
        r = sess.get(result_url, timeout=self._timeout_s)
        r.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(r.content)
        return out_path
