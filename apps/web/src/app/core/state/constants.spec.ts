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
  DEFAULT_ACTIVE_ROUTE,
  DEFAULT_WIZARD_PAGE,
  DEFAULT_EXPORT_PNG_ENABLED,
  DEFAULT_EXPORT_LBRN2_ENABLED,
  DEFAULT_EXPORT_STL_ENABLED,
  LOCAL_STORAGE_DEBOUNCE_MS,
  DEFAULT_HEIGHTMAP_SETTINGS,
  cloneDefaultStudioState,
  serializeStudioState,
  deserializeStudioState,
  DEFAULT_STUDIO_STATE,
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
  HEIGHTMAP_POLARITIES,
  SIGNATURE_CORNERS,
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

  it('DEFAULT_MASK_BACKEND is threshold (zero install)', () => {
    expect(DEFAULT_MASK_BACKEND).toBe('threshold');
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

  it('DEFAULT_ACTIVE_ROUTE is wizard', () => {
    expect(DEFAULT_ACTIVE_ROUTE).toBe('wizard');
  });

  it('DEFAULT_WIZARD_PAGE is 0', () => {
    expect(DEFAULT_WIZARD_PAGE).toBe(0);
  });

  it('DEFAULT_EXPORT_PNG_ENABLED is true', () => {
    expect(DEFAULT_EXPORT_PNG_ENABLED).toBe(true);
  });

  it('DEFAULT_EXPORT_LBRN2_ENABLED is true', () => {
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
// HeightmapSettings defaults — must mirror apps/api/schemas.py:HeightmapSettings
// ---------------------------------------------------------------------------

describe('DEFAULT_HEIGHTMAP_SETTINGS', () => {
  it('input prep toggles default off', () => {
    expect(DEFAULT_HEIGHTMAP_SETTINGS.input_white_balance).toBe(false);
    expect(DEFAULT_HEIGHTMAP_SETTINGS.input_clahe).toBe(false);
    expect(DEFAULT_HEIGHTMAP_SETTINGS.input_denoise).toBe(false);
    expect(DEFAULT_HEIGHTMAP_SETTINGS.input_remove_specular).toBe(false);
  });

  it('external heightmap polarity defaults to bright_raised (sculptok)', () => {
    expect(DEFAULT_HEIGHTMAP_SETTINGS.external_heightmap_polarity).toBe('bright_raised');
  });

  it('polarity_invert defaults off (no-signet-ring)', () => {
    expect(DEFAULT_HEIGHTMAP_SETTINGS.polarity_invert).toBe(false);
  });

  it('LightBurn convention defaults: black_is_deep + background_value=1', () => {
    expect(DEFAULT_HEIGHTMAP_SETTINGS.black_is_deep).toBe(true);
    expect(DEFAULT_HEIGHTMAP_SETTINGS.background_value).toBe(1.0);
  });

  it('refinement passes default off', () => {
    expect(DEFAULT_HEIGHTMAP_SETTINGS.subject_mask_enabled).toBe(false);
    expect(DEFAULT_HEIGHTMAP_SETTINGS.pre_clean_enabled).toBe(false);
    expect(DEFAULT_HEIGHTMAP_SETTINGS.photo_tonal_enabled).toBe(false);
  });

  it('signature text starts empty', () => {
    expect(DEFAULT_HEIGHTMAP_SETTINGS.signature_text).toBe('');
    expect(DEFAULT_HEIGHTMAP_SETTINGS.signature_corner).toBe('br');
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
    expect(WIZARD_PAGE_LABELS[2]).toBe('3. Prep & Refine');
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

  it('WIZARD_MASK_BACKENDS lists birefnet / rembg / threshold', () => {
    const values = WIZARD_MASK_BACKENDS.map((b) => b.value);
    expect(values).toContain('birefnet');
    expect(values).toContain('rembg');
    expect(values).toContain('threshold');
  });
});

// ---------------------------------------------------------------------------
// studio-shell constants
// ---------------------------------------------------------------------------

describe('studio-shell constants', () => {
  it('STUDIO_SECTION_TITLES carries the new sculptok-only sections', () => {
    expect(STUDIO_SECTION_TITLES).toHaveProperty('mask');
    expect(STUDIO_SECTION_TITLES).toHaveProperty('input');
    expect(STUDIO_SECTION_TITLES).toHaveProperty('render');
    expect(STUDIO_SECTION_TITLES).toHaveProperty('heightmap');
    expect(STUDIO_SECTION_TITLES).toHaveProperty('refinement');
    expect(STUDIO_SECTION_TITLES).toHaveProperty('output');
  });

  it('STUDIO_MASK_BACKENDS lists birefnet / rembg / threshold', () => {
    const values = STUDIO_MASK_BACKENDS.map((b) => b.value);
    expect(values).toContain('birefnet');
    expect(values).toContain('rembg');
    expect(values).toContain('threshold');
  });

  it('HEIGHTMAP_POLARITIES covers the three sculptok polarity modes', () => {
    const values = HEIGHTMAP_POLARITIES.map((o) => o.value);
    expect(values).toContain('bright_raised');
    expect(values).toContain('dark_raised');
    expect(values).toContain('auto');
  });

  it('SIGNATURE_CORNERS covers all four corners', () => {
    const values = SIGNATURE_CORNERS.map((o) => o.value);
    expect(values).toEqual(expect.arrayContaining(['tl', 'tr', 'bl', 'br']));
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
    const partial = JSON.stringify({ ui: { wizardPage: 2 } });
    const state = deserializeStudioState(partial);
    expect(state.ui.wizardPage).toBe(2);
    expect(state.session).toBeDefined();
    expect(state.pipeline).toBeDefined();
  });

  it('deserializeStudioState merges partial pipeline.settings with defaults', () => {
    const partial = JSON.stringify({
      pipeline: { settings: { input_clahe: true } },
    });
    const state = deserializeStudioState(partial);
    expect(state.pipeline.settings.input_clahe).toBe(true);
    // Other settings keys must still come from DEFAULT_HEIGHTMAP_SETTINGS.
    expect(state.pipeline.settings.input_denoise).toBe(false);
  });
});
