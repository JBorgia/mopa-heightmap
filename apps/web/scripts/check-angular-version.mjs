/**
 * CI guard: verifies the installed @angular/core major version matches
 * the pinned major declared here.  Run via: node scripts/check-angular-version.mjs
 */

import { createRequire } from 'module';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const REQUIRED_MAJOR = 21;

const __dirname = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);

const pkgPath = join(__dirname, '..', 'node_modules', '@angular', 'core', 'package.json');
const { version } = require(pkgPath);

const [major] = version.split('.').map(Number);

if (major !== REQUIRED_MAJOR) {
  console.error(
    `[check-angular-version] FAIL: expected @angular/core major ${REQUIRED_MAJOR}, ` +
    `found ${version}. Update REQUIRED_MAJOR in this script after a deliberate upgrade.`
  );
  process.exit(1);
}

console.log(`[check-angular-version] OK: @angular/core ${version} (major ${major})`);
