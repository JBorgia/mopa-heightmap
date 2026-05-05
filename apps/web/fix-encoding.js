const fs = require('fs');
const path = 'src/app/features/wizard/wizard-shell.component.ts';
let c = fs.readFileSync(path, 'utf8');

// Exact Unicode code points from inspection (Windows-1252 double-encoding mojibake)
// [garbled sequence code points, intended character]
const fixes = [
  // â€¦ (U+00E2 U+20AC U+00A6) -> … (U+2026 HORIZONTAL ELLIPSIS)
  ['\u00e2\u20ac\u00a6', '\u2026'],
  // Ã— (U+00C3 U+2014) -> × (U+00D7 MULTIPLICATION SIGN)
  ['\u00c3\u2014', '\u00d7'],
  // â‰  (U+00E2 U+2030 U+00A0) -> ≠ (U+2260 NOT EQUAL TO)
  ['\u00e2\u2030\u00a0', '\u2260'],
  // â€" (U+00E2 U+20AC U+201D) -> — (U+2014 EM DASH)
  ['\u00e2\u20ac\u201d', '\u2014'],
  // â€" (U+00E2 U+20AC U+201C) -> – (U+2013 EN DASH)
  ['\u00e2\u20ac\u201c', '\u2013'],
  // â†→ (U+00E2 U+2020 U+0090) -> ← (U+2190 LEFT ARROW)
  ['\u00e2\u2020\u0090', '\u2190'],
  // â†' (U+00E2 U+2020 U+2019) -> → (U+2192 RIGHT ARROW)
  ['\u00e2\u2020\u2019', '\u2192'],
];

let count = 0;
for (const [bad, good] of fixes) {
  const before = c;
  c = c.split(bad).join(good);
  const n = (before.length - c.length) / (bad.length - good.length);
  if (!isNaN(n) && n > 0) {
    count += Math.round(n);
    console.log(`Replaced ${Math.round(n)}x "${bad}" -> "${good}"`);
  }
}

fs.writeFileSync(path, c, 'utf8');
console.log(`Done. ${count} total replacements.`);
