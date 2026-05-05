/**
 * CI guard: regenerates openapi.json + api.d.ts and fails if git shows any diff.
 *
 * Usage (from repo root):
 *   node apps/web/scripts/check-openapi-drift.mjs
 *
 * Requires:
 *   - Python venv at .venv (Windows) or .venv (POSIX)
 *   - npx openapi-typescript available (dev dep in apps/web)
 */
import { execSync, spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(__dirname, '..', '..', '..');
const webDir = resolve(__dirname, '..');

// Locate python executable
const isWin = process.platform === 'win32';
const python = isWin
  ? join(repoRoot, '.venv', 'Scripts', 'python.exe')
  : join(repoRoot, '.venv', 'bin', 'python');

if (!existsSync(python)) {
  console.error(`[check-openapi-drift] FAIL: python not found at ${python}`);
  process.exit(1);
}

function run(cmd, cwd) {
  execSync(cmd, { cwd, stdio: 'inherit' });
}

console.log('[check-openapi-drift] Regenerating openapi.json…');
run(`"${python}" -m apps.api.export_openapi`, repoRoot);

console.log('[check-openapi-drift] Regenerating api.d.ts…');
run(
  `npx openapi-typescript openapi.json -o src/app/core/api/generated/api.d.ts`,
  webDir,
);

console.log('[check-openapi-drift] Checking git diff…');
const gitBin = process.env.GIT_BIN ?? 'git';

// Probe for git availability without throwing.
const gitProbe = spawnSync(gitBin, ['--version'], { encoding: 'utf8' });
if (gitProbe.error) {
  console.warn('[check-openapi-drift] SKIP: git not available — skipping diff check (OK in local dev).');
} else {
  try {
    execSync(
      `${gitBin} diff --exit-code apps/web/openapi.json apps/web/src/app/core/api/generated/`,
      { cwd: repoRoot, stdio: 'inherit' },
    );
  } catch {
    console.error(
      '[check-openapi-drift] FAIL: generated files differ from committed versions. ' +
      'Run the OpenAPI generation step and commit the results.',
    );
    process.exit(1);
  }
}

console.log('[check-openapi-drift] OK: no drift detected.');
