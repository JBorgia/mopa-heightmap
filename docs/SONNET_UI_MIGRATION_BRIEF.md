# Sonnet handoff — Phase 9 UI/UX migration brief

> Self-contained brief for a Sonnet-class agent to take the MOPA Heightmap
> Studio from its current Gradio UI to an Angular + PrimeNG + SignalTree
> SPA backed by a FastAPI service. Read `IMPLEMENTATION_PLAN.md` §9b for
> the architectural rationale; this file is the operational plan.

---

## 0. Context (load this first)

- **Repo:** `c:\Users\TwentyOne21\code\mopa-heightmap`
- **Python venv:** `.\.venv\Scripts\python.exe` (Python 3.11.15)
- **Current UI entry point:** `python -m ui.mopa_studio` → http://127.0.0.1:7860
- **Tests:** `pytest` (326/326 green at handoff). Wizard-specific suites:
  `tests/test_wizard.py` + `tests/test_wizard_phase8.py` (36 tests).
- **Hardware target:** 60 W M7 MOPA, Quadro P2000 4 GB, Windows 10/11.
- **License policy:** Apache-2 / MIT default; CC-BY-NC opt-in only via
  `InferenceSettings.allow_nc_weights`.
- **Standing constraints (DO NOT violate):**
  - Every magic value must be a named module-level constant exported and pinned by a test.
  - Heightmap PNG output is always-on; `.lbrn2` checkbox defaults ON.
  - Reactive runner uses hash-based caching (`InferenceService.image_hash`).
  - Click-refine is intentionally NOT shipped in Studio (slider UX); it lives only in the wizard.
  - Do NOT update `IMPLEMENTATION_PLAN.md` unless explicitly asked.

---

## 1. What "done" looks like

A Sonnet implementer is finished when:

1. `apps/api/` runs FastAPI on `:8000`, exposes the headless service surface (§3),
   and **has 100 % parity** with `InferenceService` semantics — same hashes,
   same outputs, same defaults.
2. `apps/web/` is an Angular workspace that boots PrimeNG (Aura), uses
   SignalTree as the single root state container, and reproduces the wizard's
   five pages **and** the studio's full pass stack.
3. `python -m ui.mopa_studio` (Gradio) and the new SPA can both be run; they
   produce **byte-identical** heightmap PNGs for the same input + settings
   (golden-file test in `tests/test_parity_gradio_vs_spa.py`).
4. Performance budgets in §6 are enforced by Playwright + pytest-benchmark.
5. All 326 existing pytest tests still pass; new tests bring the count up
   without lowering coverage.

The cutover (Phase 9f) **deletes** `ui/mopa_studio.py` and `ui/mopa_wizard.py`
in the same PR that flips the README to point at the SPA.

---

## 2. Repository layout after migration

```
mopa-heightmap/
├── apps/
│   ├── api/                      # FastAPI service (was ui/mopa_studio.py)
│   │   ├── main.py               # uvicorn entrypoint
│   │   ├── routes/
│   │   │   ├── render.py         # POST /render, /render/stream
│   │   │   ├── mask.py           # POST /mask, /mask/click
│   │   │   ├── profile.py        # CRUD for profiles
│   │   │   ├── export.py         # /export/png, /export/lbrn2, /export/stl
│   │   │   └── session.py        # WebSocket /ws/session/{id}
│   │   ├── schemas.py            # Pydantic models (mirror InferenceSettings)
│   │   └── service_adapter.py    # Wraps existing zoedepth.laser.* code
│   └── web/                      # Angular workspace
│       ├── angular.json
│       ├── package.json
│       ├── jest.config.ts
│       ├── playwright.config.ts
│       └── src/
│           ├── app/
│           │   ├── app.config.ts                 # standalone bootstrap
│           │   ├── app.routes.ts                 # /wizard /studio /export
│           │   ├── core/
│           │   │   ├── state/                    # SignalTree definitions
│           │   │   ├── api/                      # typed http clients
│           │   │   └── transport/                # binary blob helpers, hashing
│           │   ├── features/
│           │   │   ├── wizard/                   # 5 pages, lazy-loaded
│           │   │   ├── studio/                   # accordion-driven full UI
│           │   │   └── export/                   # PNG / .lbrn2 / .stl
│           │   └── shared/                       # PrimeNG wrappers
│           └── styles/
├── ui/                           # DELETED in Phase 9f
├── zoedepth/laser/               # unchanged — single source of truth
├── tests/                        # existing pytest suite + new parity tests
└── IMPLEMENTATION_PLAN.md
```

---

## 3. Headless service surface (Phase 9a — do this first)

**The single rule:** the API must be a *thin* adapter. All math stays in
`zoedepth/laser/*` and `ui/services/*`. No business logic in `apps/api/routes/`.

### Endpoint contract

| Method | Path | Body | Returns | Notes |
|---|---|---|---|---|
| POST | `/upload` | multipart `image` | `{image_id, w, h, sha256}` | Stored in temp dir keyed by sha256 |
| POST | `/render` | `RenderRequest` JSON + `image_id` | `{heightmap_id, preview_id, elapsed_s, image_hash}` | Synchronous |
| GET  | `/render/stream/{job_id}` | — | SSE: `{progress, partial_id?}` | For multires + Phase 8 passes |
| POST | `/mask` | `{image_id, backend, edge_softness}` | `{mask_id, coverage_pct}` | |
| POST | `/mask/click` | `{image_id, mask_id, x, y, label, tol, max_frac}` | `{mask_id, coverage_pct}` | |
| GET  | `/blob/{id}` | — | `image/png` (binary) | `Cache-Control: public, max-age=31536000, immutable` |
| GET  | `/profiles` / `POST` `/profiles` | — / Profile JSON | List / created profile | |
| POST | `/plan` | `PassPlanRequest` | `PassPlan` | Wraps `plan_passes()` |
| POST | `/export/png` | `{heightmap_id, settings}` | `image/png` 16-bit | |
| POST | `/export/lbrn2` | `{plan_id, settings}` | `application/xml` | |
| POST | `/export/stl` | `{heightmap_id, settings}` | `model/stl` | Phase 8 |
| WS   | `/ws/session/{id}` | — | `{event, payload}` | Client mirror of SignalTree slice |

### Pydantic schemas

Mirror `InferenceSettings` field-for-field. Generate TypeScript types into
`apps/web/src/app/core/api/generated/` via `datamodel-code-generator` or
`openapi-typescript` so drift is impossible. Add a CI step:

```powershell
.\.venv\Scripts\python.exe -m apps.api.export_openapi > apps/web/openapi.json
npx openapi-typescript apps/web/openapi.json -o apps/web/src/app/core/api/generated/api.d.ts
git diff --exit-code apps/web/src/app/core/api/generated/  # fail if stale
```

### Parity test (must exist before any UI work)

`tests/test_api_parity.py` — for a fixed input image + fixed settings, assert
the bytes of `/export/png` equal the bytes produced by calling
`InferenceService.render(...)` directly + the existing PNG writer.

---

## 4. Angular workspace bootstrap (Phase 9b)

```powershell
cd apps
npm i -g @angular/cli@latest
ng new web --standalone --routing --style=scss --strict --package-manager=npm
cd web
npm i primeng primeicons primeflex
npm i @angular/animations
npm i -D jest @types/jest jest-preset-angular @playwright/test
# SignalTree: vendored copy under apps/web/src/lib/signal-tree/
```

### `app.config.ts` essentials

- `provideRouter` with lazy routes for `/wizard` and `/studio`.
- `provideAnimations()` (PrimeNG requires it).
- `providePrimeNG({ theme: { preset: Aura, options: { darkModeSelector: '.dark' } } })`.
- `provideHttpClient(withFetch())`.
- Provide a global `SessionTreeService` that owns the root SignalTree.

### CI guard (required)

Add `apps/web/scripts/check-angular-version.mjs`:

```js
import { execSync } from 'node:child_process';
import pkg from '../package.json' assert { type: 'json' };
const latest = execSync('npm view @angular/core version').toString().trim();
const ours = pkg.dependencies['@angular/core'].replace(/^[^0-9]*/, '');
const [lm] = latest.split('.'); const [om] = ours.split('.');
if (Number(latest.split('.')[0]) - Number(ours.split('.')[0]) > 0) {
  console.error(`Angular ${ours} is behind ${latest}`); process.exit(1);
}
```

Run it in CI: `node scripts/check-angular-version.mjs`.

---

## 5. SignalTree shape (Phase 9c)

Single root, four top-level slices. Keep the tree shallow; large binary data
(Float32Arrays) lives in plain refs and only the **content hash** is signaled.

```ts
export interface StudioState {
  session: {
    imageId: string | null;
    imageHash: string | null;             // signaled
    sourceMeta: { w: number; h: number; bytes: number } | null;
    history: HistoryEntry[];              // last 20 actions
  };
  pipeline: {
    mask: { backend: MaskBackend; edgeSoftness: number; maskId: string | null; coveragePct: number };
    render: { detailBalance: number; multires: boolean; relief: number; profileName: string | null };
    advanced: {                           // Phase 8 accordion
      preUpscale: boolean; upscaler: 'realesrgan'|'swinir'; targetMP: number;
      sharpen: number; toneCurve: ToneCurvePoint[]; bilateralStrength: number;
      // … one field per STUDIO_DEFAULT_* constant
    };
  };
  output: {
    heightmapId: string | null;           // signaled — drives preview
    previewId: string | null;
    plan: PassPlan | null;
    elapsedSeconds: number | null;
  };
  ui: {                                   // pure client state, never sent to API
    activeRoute: 'wizard'|'studio'|'export';
    wizardPage: 0|1|2|3|4;
    rightPaneCollapsed: boolean;
    toasts: Toast[];
  };
}
```

### Wiring rules

- **One service per top-level slice** — `MaskService`, `RenderService`, `ExportService`. Each owns its sub-tree and exposes `update*` methods + `Observable`-shaped derivations.
- **No two-way binding to SignalTree from templates.** Templates read via `tree.pipeline.render.detailBalance()` and dispatch via `service.setDetailBalance(v)`.
- **Side-effect boundary:** API calls only inside services. Components never call `HttpClient`.
- **Effect for autosave:** an `effect()` in `SessionTreeService` writes the tree (minus binary refs) to `localStorage` on every change, debounced 500 ms.

---

## 6. Performance budgets (enforced)

Add `apps/web/playwright/perf.spec.ts` with hard assertions:

| Budget | Limit | Test approach |
|---|---|---|
| Time-to-interactive on local load | < 800 ms | Playwright `performance.timing` |
| Slider drag → cached preview repaint | < 16 ms | `requestAnimationFrame` timing |
| `/render` round-trip overhead vs direct Python call | < 50 ms | pytest-benchmark in `tests/test_api_parity.py` |
| 2048² 16-bit PNG transport on loopback | < 80 ms | `fetch()` + `performance.now()` |
| First contentful paint of `/wizard` | < 400 ms | Lighthouse CI |
| Bundle size (initial chunk gzip) | < 300 KB | `ng build --stats-json` + `bundlesize` |

**A failing budget fails CI.** No exceptions; tune the code, not the budget.

---

## 7. Phased execution (do strictly in this order)

### Phase 9a — Headless service (1–2 days)
1. Create `apps/api/` skeleton + `service_adapter.py` wrapping `InferenceService`.
2. Implement `/upload`, `/render`, `/mask`, `/blob`, `/export/png` — minimal viable surface.
3. Write `tests/test_api_parity.py` — must pass before continuing.
4. Generate `openapi.json` + `api.d.ts`. Wire CI check.

### Phase 9b — Angular bootstrap (½ day)
1. `ng new` per §4. Commit clean baseline.
2. Add PrimeNG, SignalTree, Jest, Playwright config.
3. Add Angular-version CI guard.
4. Implement empty `/wizard` and `/studio` routes that render placeholder PrimeNG cards.

### Phase 9c — SignalTree + binary transport (1 day)
1. Implement the tree from §5 with full TypeScript types.
2. Build `BlobCache` service: `Map<string, Blob>` keyed by content hash, with LRU eviction at 200 MB.
3. Build typed HTTP client per route, fully mocked for unit tests.
4. Add `localStorage` autosave effect.

### Phase 9d — Wizard parity (2 days)
1. Five lazy-loaded routes mirroring `_WIZARD_PAGE_LABELS` in `ui/mopa_wizard.py`.
2. Persistent right pane (`<p-splitter>`) with original + preview thumbnails — fed by SignalTree, no `.then()` chains needed.
3. Each page action calls a service which calls the API. **Reuse the same backend keys** (`birefnet`, `rembg`, `flood-fill`, etc).
4. Port the click-refine UX with PrimeNG `<p-image>` overlay click handler. **Bug fix from current Gradio:** the click handler must use a *clicker* registry key, never the *mask backend* key (see §10).
5. Snapshot tests against existing wizard golden outputs.

### Phase 9e — Studio parity + Phase 8 advanced accordion (2 days)
1. Single `/studio` route with PrimeNG `<p-accordion>` containing all `STUDIO_DEFAULT_*` controls.
2. Reactive preview: any `update*` call hashes inputs and only re-renders if the hash changes (mirror current `InferenceService` cache).
3. Toast confirmations for export actions.
4. Drag-and-drop upload via PrimeNG `<p-fileUpload mode="advanced">` with chunked transfer.

### Phase 9f — Cutover (½ day)
1. Update README: launch command becomes `uvicorn apps.api.main:app` + serve `apps/web/dist/` as static.
2. Delete `ui/mopa_studio.py`, `ui/mopa_wizard.py`, and their tests (move parity coverage to API + Playwright).
3. Add a single integration smoke test that boots both servers and runs a full wizard flow end-to-end.
4. Tag the commit `v9.0-spa`.

---

## 8. Testing strategy

| Layer | Tool | What it pins |
|---|---|---|
| Python service | pytest (existing 326) | All zoedepth.laser + ui.services contracts |
| API parity | pytest + httpx | Endpoint outputs == direct service calls (byte-exact) |
| Angular unit | Jest | Components in isolation, services with mocked HTTP |
| Angular integration | Jest + Angular TestBed | SignalTree mutations propagate correctly |
| E2E | Playwright | User flows + perf budgets |
| Golden parity | pytest | Same image + settings → same PNG bytes from old + new path |
| Bundle size | bundlesize | Initial chunk < 300 KB gzip |
| Lighthouse | Lighthouse CI | FCP, TTI, accessibility ≥ 95 |

**Coverage gate:** new TS code ≥ 85 % statement coverage. New Python ≥ 90 %.

---

## 9. Known traps (do not relearn these the hard way)

1. **HF cache symlink warning on Windows** is harmless; do NOT enable Developer Mode in CI.
2. **Gradio's queue locks on unhandled handler exceptions** — irrelevant once Gradio is gone, but the API must still return JSON errors with a stable shape: `{error: {code, message, hint?}}`.
3. **`einops` + `kornia`** are required by BiRefNet; pin them in `pyproject.toml` so a fresh clone works.
4. **PIL Image transport across processes** is now HTTP — never base64-encode. Always `multipart/form-data` upload, `image/png` (or `image/webp` for previews) download.
5. **SignalTree + huge typed arrays:** putting a 2048² Float32Array in a signal causes Angular to attempt structural equality on every check. Store the array in a plain `Map`, signal only the hash.
6. **PrimeNG slider `pInputNumber` + reactive forms** has historically had an event-emission timing bug on certain versions; prefer `(onSlideEnd)` over `(onChange)` for expensive recomputes (matches current Gradio `.release()` semantics).
7. **CORS in dev:** Angular dev-server runs on `:4200`, FastAPI on `:8000`. Add `CORSMiddleware` allowing `http://localhost:4200` only in dev mode (env-gated).

---

## 10. Bug ledger from current Gradio UI (fix during port; do not re-introduce)

| ID | Symptom | Root cause | Fix in SPA |
|---|---|---|---|
| BUG-1 | "Page can't refresh" | Unhandled handler exception locks Gradio queue | API returns proper HTTP errors; UI shows toast |
| BUG-2 | `KeyError: 'birefnet'` on click-refine | Click handler was wired to mask backend dropdown instead of a clicker key | Separate `MaskBackend` and `ClickerKey` types in TS; never share a control |
| BUG-3 | Whole-page scroll on laptops | Gradio root container ignored viewport height | PrimeNG splitter + `height: 100dvh` works natively |
| BUG-4 | Preview state lost across tabs | Tab-level Gradio state did not propagate | SignalTree is global; tabs are pure projections |
| BUG-5 | Stale port 7860 leaks across runs | Gradio doesn't release socket on Ctrl+C | uvicorn handles this; add `--reload` for dev |

---

## 11. Acceptance checklist (Sonnet ticks these before declaring done)

- [ ] `apps/api/` runs and `pytest tests/test_api_parity.py` passes.
- [ ] All 326 legacy pytest tests still green.
- [ ] `apps/web/` builds with `ng build --configuration=production` with zero warnings.
- [ ] `npx jest` passes with ≥ 85 % statement coverage on new code.
- [ ] `npx playwright test` passes including all perf budgets in §6.
- [ ] Angular-version CI guard runs and is green.
- [ ] OpenAPI → TS generation runs in CI and `git diff --exit-code` is clean.
- [ ] Manual smoke: upload image → run wizard → export PNG → export `.lbrn2` → export `.stl`. Bytes match Gradio output.
- [ ] README updated with new launch instructions.
- [ ] `ui/mopa_studio.py` and `ui/mopa_wizard.py` deleted in the cutover commit.

---

## 12. Out of scope (explicitly NOT in Phase 9)

- New ML models (depth backbones, segmentation networks). Phase 9 is a UI port only.
- Multi-user / multi-session. Local single-user only.
- Cloud deploy / Docker. Local execution remains the target.
- Mobile responsive beyond "laptop in landscape". Tablet/phone not supported.
- i18n. English only.
- Authentication. Localhost only; no auth layer.

---

## 13. Files Sonnet should read before starting

In order of priority:

1. `IMPLEMENTATION_PLAN.md` §9b — architectural rationale.
2. `ui/services/inference_service.py` — the contract the API must wrap.
3. `ui/mopa_wizard.py` — wizard UX to reproduce.
4. `ui/mopa_studio.py` — studio UX + Phase 8 accordion to reproduce.
5. `zoedepth/laser/subject_mask.py`, `click_mask.py`, `depth_fusion.py` — registries Sonnet must expose 1:1.
6. `tests/test_studio_phase8.py`, `tests/test_wizard_phase8.py` — golden-file patterns to mirror.
7. `profiles/*.json` — schema the `/profiles` endpoint must round-trip.

---

*End of brief. Sonnet: start with Phase 9a step 1. Do not skip the parity test.*
