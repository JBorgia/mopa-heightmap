---
name: Node.js + pnpm + Angular CLI toolchain on this machine
description: Where Node.js / pnpm / Angular CLI live on the user's Windows box, and how to invoke them from the bash/PowerShell tool which inherits a minimal PATH.
type: reference
---

The user has Node.js installed but not on the bash/PowerShell PATH that
the agent's shells inherit. Plain ``node`` / ``npm`` / ``pnpm`` / ``ng``
all return "command not found" until you re-export PATH. Concretely:

| Tool | Location |
|---|---|
| Node.js + npm + corepack | ``C:\Program Files\nodejs\`` |
| npm globals (legacy) | ``C:\Users\TwentyOne21\AppData\Roaming\npm`` |
| pnpm + pnpm globals | ``C:\Users\TwentyOne21\AppData\Local\pnpm`` |

Use this preamble at the start of any Node-toolchain bash command:

```
export PATH="/c/Users/TwentyOne21/AppData/Local/pnpm:/c/Users/TwentyOne21/AppData/Roaming/npm:/c/Program Files/nodejs:$PATH"
export PNPM_HOME="/c/Users/TwentyOne21/AppData/Local/pnpm"
```

Versions installed (2026-05-05):
- Node.js 24.15.0
- npm 11.12.1 (bundled with Node)
- pnpm 10.33.3 (installed via ``npm install -g pnpm``; corepack-shim
  enable failed because writing to ``C:\Program Files\nodejs`` needs
  admin)
- @angular/cli 21.2.9 (installed via ``pnpm add -g @angular/cli``)

apps/web is wired to **pnpm** as of commit ``<rename + pnpm migration>``:
- ``packageManager: "pnpm@10.33.3"`` in package.json
- ``pnpm-lock.yaml`` is the lockfile (``package-lock.json`` deleted)
- script references in package.json use ``pnpm run <script>``
- ``pnpm.onlyBuiltDependencies`` whitelist allows native postinstalls
  for esbuild / @parcel/watcher / lmdb / msgpackr-extract / unrs-resolver

Build / dev commands (all from ``apps/web/``):
- ``pnpm install`` — install/sync lockfile
- ``pnpm run build`` — production build (writes to ``dist/web/``)
- ``pnpm start`` — ``ng serve`` dev server
- ``pnpm test`` — vitest test runner
- ``pnpm e2e`` — Playwright end-to-end

If pnpm setup ran (it did on 2026-05-05), the system PATH for new
terminals already has these dirs prepended. Existing terminals
(including the agent's) still need the explicit export.
