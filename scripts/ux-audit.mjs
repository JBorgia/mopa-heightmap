import { chromium } from 'playwright';
import { mkdirSync } from 'fs';
import { join } from 'path';

const OUT = 'scripts/ux-screenshots';
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
await page.setViewportSize({ width: 1440, height: 900 });

async function shot(name) {
  await page.screenshot({ path: join(OUT, `${name}.png`) });
  console.log(`Screenshot: ${name}.png`);
}

async function txt(sel) {
  try { return (await page.locator(sel).first().innerText({ timeout: 3000 })).trim(); }
  catch { return '[not found]'; }
}

async function visible(sel) {
  try { return await page.locator(sel).first().isVisible({ timeout: 2000 }); }
  catch { return false; }
}

await page.goto('http://localhost:4200', { waitUntil: 'networkidle' });
await page.waitForTimeout(1000);

// ── 1. Initial load ───────────────────────────────────────────────────
await shot('01-initial-load');

// ── 2. Step 0 — no selection ──────────────────────────────────────────
console.log('\n── STEP 0: Blank Setup ──');
const chips = await page.locator('.target-chip').all();
console.log(`Chips: ${chips.length}`);
for (const c of chips) {
  console.log(' ', (await c.innerText({ timeout: 3000 })).trim().replace(/\n/g, ' | '));
}
await shot('02-step0-no-selection');

// ── 3. Click Coin ────────────────────────────────────────────────────
await page.locator('.target-chip').first().click();
await page.waitForTimeout(500);
await shot('03-step0-coin-selected');
console.log('Depth style visible:', await visible('#wiz-enhance-p0'));
console.log('Shape chips:', (await page.locator('.shape-chip').all()).length);

// ── 4. Click Signet ring (last chip, inverted) ─────────────────────
await page.locator('.target-chip').last().click();
await page.waitForTimeout(500);
await shot('04-step0-signet-inverted');
console.log('Stage badge text:', await txt('.stage-badge'));

// ── 5. Back to Coin, go to step 1 ─────────────────────────────────
await page.locator('.target-chip').first().click();
await page.waitForTimeout(300);

// Click step 1 in the page strip
const strip = await page.locator('.wizard-page-strip button').all();
console.log(`\nPage strip buttons: ${strip.length}`);
for (const s of strip) {
  console.log(' ', (await s.innerText({ timeout: 2000 })).trim().replace(/\n/g, ' | '));
}

if (strip.length > 1) {
  await strip[1].click();
  await page.waitForTimeout(600);
}
await shot('05-step1-canvas-no-photo');

// ── 6. Inspect canvas step ────────────────────────────────────────
console.log('\n── STEP 1: Canvas ──');
console.log('Canvas visible:', await visible('canvas.shape-canvas'));
const box = await page.locator('canvas.shape-canvas').first().boundingBox().catch(() => null);
console.log('Canvas bounding box:', JSON.stringify(box));
console.log('Canvas header:', await txt('.shape-canvas-header'));
console.log('Drop overlay visible:', await visible('.canvas-drop-overlay'));
console.log('File input count:', (await page.locator('input[type=file]').all()).length);
console.log('Background dropdown visible:', await visible('#wiz-bg-pattern'));

// Screenshot left panel only
await page.locator('.wizard-main').screenshot({ path: join(OUT, '05b-step1-leftpanel.png') }).catch(() => {});
console.log('Screenshot: 05b-step1-leftpanel.png');

// ── 7. Step 2 — Image Prep ────────────────────────────────────────
if (strip.length > 2) { await strip[2].click(); await page.waitForTimeout(500); }
await shot('06-step2-image-prep');
console.log('\n── STEP 2: Image Prep ──');
const controls = await page.locator('.control-group label').allInnerTexts().catch(() => []);
controls.forEach((c, i) => console.log(`  control ${i}: ${c.trim().slice(0, 60)}`));

// ── 8. Step 3 — Checkpoint ───────────────────────────────────────
if (strip.length > 3) { await strip[3].click(); await page.waitForTimeout(500); }
await shot('07-step3-checkpoint');
console.log('\n── STEP 3: Checkpoint ──');
console.log(await txt('.wizard-current-page'));

// ── 9. Step 4 — Depth Map ────────────────────────────────────────
if (strip.length > 4) { await strip[4].click(); await page.waitForTimeout(500); }
await shot('08-step4-depth-map');
console.log('\n── STEP 4: Depth Map ──');
console.log((await txt('.wizard-current-page')).slice(0, 400));

// ── 10. Step 5 — Subject & Zones ─────────────────────────────────
if (strip.length > 5) { await strip[5].click(); await page.waitForTimeout(500); }
await shot('09-step5-zones');

// ── 11. Step 7 — Export ──────────────────────────────────────────
if (strip.length > 7) { await strip[7].click(); await page.waitForTimeout(500); }
await shot('10-step7-export');
console.log('\n── STEP 7: Export ──');
console.log((await txt('.wizard-current-page')).slice(0, 400));

// ── 12. Right sidepane ───────────────────────────────────────────
if (strip.length > 0) { await strip[0].click(); await page.waitForTimeout(300); }
await page.locator('.target-chip').first().click();
await page.waitForTimeout(300);
const sidepane = page.locator('.wizard-sidepane');
if (await sidepane.isVisible().catch(() => false)) {
  await sidepane.screenshot({ path: join(OUT, '11-right-sidepane.png') });
  console.log('\nScreenshot: 11-right-sidepane.png');
}

await browser.close();
console.log('\nDone. Screenshots in', OUT);
