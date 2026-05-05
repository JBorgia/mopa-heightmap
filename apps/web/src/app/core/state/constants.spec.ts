/**
 * Unit tests for named constants across the Angular workspace.
 *
 * Per the brief: every magic value must be a named module-level constant
 * exported and pinned by a test.
 */
import { describe, it, expect } from 'vitest';

import {
  STUDIO_STATE_STORAGE_KEY,
  STUDIO_HISTORY_LIMIT,
  BLOB_CACHE_MAX_BYTES,
  DEFAULT_MASK_BACKEND,
  DEFAULT_CLICKER_KEY,
  DEFAULT_MASK_EDGE_SOFTNESS,
  DEFAULT_MASK_COVERAGE_PCT,
  DEFAULT_DETAIL_BALANCE,
  DEFAULT_RELIEF,
  DEFAULT_TARGET_MP,
  DEFAULT_SHARPEN,
  DEFAULT_BILATERAL_STRENGTH,
  DEFAULT_ACTIVE_ROUTE,
  DEFAULT_WIZARD_PAGE,
  DEFAULT_EXPORT_PNG_ENABLED,
  DEFAULT_EXPORT_LBRN2_ENABLED,
  DEFAULT_EXPORT_STL_ENABLED,
  LOCAL_STORAGE_DEBOUNCE_MS,
} from './studio-state';

import {
  WIZARD_PAGE_LABELS,
  WIZARD_STAGE_SUMMARIES,
  WIZARD_DEFAULT_SPLITTER_SIZES,
  WIZARD_COLLAPSED_SPLITTER_SIZES,
  WIZARD_HISTORY_PREVIEW_LIMIT,
  WIZARD_MASK_BACKENDS,
} from '../../features/wizard/wizard-shell.component';

import {
  STUDIO_SECTION_TITLES,
  STUDIO_MASK_BACKENDS,
  STUDIO_UPSCALER_OPTIONS,
} from '../../features/studio/studio-shell.component';

import {
  EXPORT_PNG_FILENAME,
  EXPORT_LBRN2_FILENAME,
  EXPORT_STL_FILENAME,
  EXPORT_STL_DEFAULT_Z_SCALE_MM,
  EXPORT_STL_DEFAULT_BASE_THICKNESS_MM,
} from './export.service';

import { API_BASE_URL } from '../api/api-client.service';

// ---------------------------------------------------------------------------
// studio-state constants
// ---------------------------------------------------------------------------

describe('studio-state constants', () => {
  it('STUDIO_STATE_STORAGE_KEY is stable', () => {
    expect(STUDIO_STATE_STORAGE_KEY).toBe('mopa-heightmap.studio-state');
  });

  it('STUDIO_HISTORY_LIMIT is 20', () => {
    expect(STUDIO_HISTORY_LIMIT).toBe(20);
  });

  it('BLOB_CACHE_MAX_BYTES is 200 MB', () => {
    expect(BLOB_CACHE_MAX_BYTES).toBe(200 * 1024 * 1024);
  });

  it('DEFAULT_MASK_BACKEND is rembg', () => {
    expect(DEFAULT_MASK_BACKEND).toBe('rembg');
  });

  it('DEFAULT_CLICKER_KEY is flood-fill', () => {
    expect(DEFAULT_CLICKER_KEY).toBe('flood-fill');
  });

  it('DEFAULT_MASK_EDGE_SOFTNESS is 0', () => {
    expect(DEFAULT_MASK_EDGE_SOFTNESS).toBe(0);
  });

  it('DEFAULT_MASK_COVERAGE_PCT is 0', () => {
    expect(DEFAULT_MASK_COVERAGE_PCT).toBe(0);
  });

  it('DEFAULT_DETAIL_BALANCE is 0.35', () => {
    expect(DEFAULT_DETAIL_BALANCE).toBeCloseTo(0.35);
  });

  it('DEFAULT_RELIEF is 1', () => {
    expect(DEFAULT_RELIEF).toBe(1);
  });

  it('DEFAULT_TARGET_MP is 2', () => {
    expect(DEFAULT_TARGET_MP).toBe(2);
  });

  it('DEFAULT_SHARPEN is 0.2', () => {
    expect(DEFAULT_SHARPEN).toBeCloseTo(0.2);
  });

  it('DEFAULT_BILATERAL_STRENGTH is 0.08', () => {
    expect(DEFAULT_BILATERAL_STRENGTH).toBeCloseTo(0.08);
  });

  it('DEFAULT_ACTIVE_ROUTE is wizard', () => {
    expect(DEFAULT_ACTIVE_ROUTE).toBe('wizard');
  });

  it('DEFAULT_WIZARD_PAGE is 0', () => {
    expect(DEFAULT_WIZARD_PAGE).toBe(0);
  });

  it('DEFAULT_EXPORT_PNG_ENABLED is true', () => {
    expect(DEFAULT_EXPORT_PNG_ENABLED).toBe(true);
  });

  it('DEFAULT_EXPORT_LBRN2_ENABLED is true (on by default per brief)', () => {
    expect(DEFAULT_EXPORT_LBRN2_ENABLED).toBe(true);
  });

  it('DEFAULT_EXPORT_STL_ENABLED is false', () => {
    expect(DEFAULT_EXPORT_STL_ENABLED).toBe(false);
  });

  it('LOCAL_STORAGE_DEBOUNCE_MS is 500', () => {
    expect(LOCAL_STORAGE_DEBOUNCE_MS).toBe(500);
  });
});

// ---------------------------------------------------------------------------
// wizard-shell constants
// ---------------------------------------------------------------------------

describe('wizard-shell constants', () => {
  it('WIZARD_PAGE_LABELS has exactly 5 entries', () => {
    expect(WIZARD_PAGE_LABELS).toHaveLength(5);
  });

  it('WIZARD_PAGE_LABELS matches expected labels in order', () => {
    expect(WIZARD_PAGE_LABELS[0]).toBe('1. Upload');
    expect(WIZARD_PAGE_LABELS[1]).toBe('2. Subject');
    expect(WIZARD_PAGE_LABELS[2]).toBe('3. Form & Detail');
    expect(WIZARD_PAGE_LABELS[3]).toBe('4. Material & Passes');
    expect(WIZARD_PAGE_LABELS[4]).toBe('5. Review & Export');
  });

  it('WIZARD_STAGE_SUMMARIES has exactly 5 entries', () => {
    expect(WIZARD_STAGE_SUMMARIES).toHaveLength(5);
  });

  it('WIZARD_DEFAULT_SPLITTER_SIZES sums to 100', () => {
    const sum = WIZARD_DEFAULT_SPLITTER_SIZES.reduce((a, b) => a + b, 0);
    expect(sum).toBe(100);
  });

  it('WIZARD_COLLAPSED_SPLITTER_SIZES collapses right pane to 0', () => {
    expect(WIZARD_COLLAPSED_SPLITTER_SIZES[1]).toBe(0);
  });

  it('WIZARD_HISTORY_PREVIEW_LIMIT is 5', () => {
    expect(WIZARD_HISTORY_PREVIEW_LIMIT).toBe(5);
  });

  it('WIZARD_MASK_BACKENDS has entries for birefnet, rembg, flood-fill', () => {
    const values = WIZARD_MASK_BACKENDS.map((b) => b.value);
    expect(values).toContain('birefnet');
    expect(values).toContain('rembg');
    expect(values).toContain('flood-fill');
  });

  it('WIZARD_MASK_BACKENDS flood-fill value matches DEFAULT_CLICKER_KEY', () => {
    // flood-fill is the only valid ClickerKey; it must appear as a backend too.
    expect(WIZARD_MASK_BACKENDS.some((b) => b.value === DEFAULT_CLICKER_KEY)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// studio-shell constants
// ---------------------------------------------------------------------------

describe('studio-shell constants', () => {
  it('STUDIO_SECTION_TITLES has mask, render, advanced, output keys', () => {
    expect(STUDIO_SECTION_TITLES).toHaveProperty('mask');
    expect(STUDIO_SECTION_TITLES).toHaveProperty('render');
    expect(STUDIO_SECTION_TITLES).toHaveProperty('advanced');
    expect(STUDIO_SECTION_TITLES).toHaveProperty('output');
  });

  it('STUDIO_MASK_BACKENDS includes all three backends', () => {
    const values = STUDIO_MASK_BACKENDS.map((b) => b.value);
    expect(values).toContain('birefnet');
    expect(values).toContain('rembg');
    expect(values).toContain('flood-fill');
  });

  it('STUDIO_UPSCALER_OPTIONS includes realesrgan and swinir', () => {
    const values = STUDIO_UPSCALER_OPTIONS.map((o: { value: string }) => o.value);
    expect(values).toContain('realesrgan');
    expect(values).toContain('swinir');
  });
});

// ---------------------------------------------------------------------------
// export.service constants
// ---------------------------------------------------------------------------

describe('export.service constants', () => {
  it('EXPORT_PNG_FILENAME ends with .png', () => {
    expect(EXPORT_PNG_FILENAME).toMatch(/\.png$/);
  });

  it('EXPORT_LBRN2_FILENAME ends with .lbrn2', () => {
    expect(EXPORT_LBRN2_FILENAME).toMatch(/\.lbrn2$/);
  });

  it('EXPORT_STL_FILENAME ends with .stl', () => {
    expect(EXPORT_STL_FILENAME).toMatch(/\.stl$/);
  });

  it('EXPORT_STL_DEFAULT_Z_SCALE_MM is positive', () => {
    expect(EXPORT_STL_DEFAULT_Z_SCALE_MM).toBeGreaterThan(0);
  });

  it('EXPORT_STL_DEFAULT_BASE_THICKNESS_MM is positive', () => {
    expect(EXPORT_STL_DEFAULT_BASE_THICKNESS_MM).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// api-client constants
// ---------------------------------------------------------------------------

describe('ApiClientService constants', () => {
  it('API_BASE_URL points to localhost:8000', () => {
    expect(API_BASE_URL).toContain(':8000');
    expect(API_BASE_URL).toMatch(/^https?:\/\//);
  });
});

// ---------------------------------------------------------------------------
// studio-state utility functions
// ---------------------------------------------------------------------------

import {
  cloneDefaultStudioState,
  serializeStudioState,
  deserializeStudioState,
  DEFAULT_STUDIO_STATE,
} from './studio-state';

describe('studio-state utilities', () => {
  it('cloneDefaultStudioState returns a deep clone (not reference)', () => {
    const a = cloneDefaultStudioState();
    const b = cloneDefaultStudioState();
    expect(a).not.toBe(b);
    expect(a).toEqual(b);
  });

  it('serializeStudioState produces valid JSON', () => {
    const json = serializeStudioState(DEFAULT_STUDIO_STATE);
    expect(() => JSON.parse(json)).not.toThrow();
  });

  it('deserializeStudioState(null) returns default state', () => {
    const state = deserializeStudioState(null);
    expect(state.session.imageId).toBe(DEFAULT_STUDIO_STATE.session.imageId);
  });

  it('deserializeStudioState(invalid JSON) returns default state', () => {
    const state = deserializeStudioState('{not: valid json}');
    expect(state.session.imageId).toBe(DEFAULT_STUDIO_STATE.session.imageId);
  });

  it('deserializeStudioState restores wizardPage from valid JSON', () => {
    const modified = cloneDefaultStudioState();
    modified.ui.wizardPage = 3 as typeof modified.ui.wizardPage;
    const json = serializeStudioState(modified);
    const restored = deserializeStudioState(json);
    expect(restored.ui.wizardPage).toBe(3);
  });

  it('deserializeStudioState merges partial state with defaults', () => {
    // Only include ui field to check merging logic
    const partial = JSON.stringify({ ui: { wizardPage: 2 } });
    const state = deserializeStudioState(partial);
    expect(state.ui.wizardPage).toBe(2);
    // Default fields should still be present
    expect(state.session).toBeDefined();
    expect(state.pipeline).toBeDefined();
  });
});
