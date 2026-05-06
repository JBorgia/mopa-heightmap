import type { HeightmapSettings } from '../api/api-types';

export const STUDIO_STATE_STORAGE_KEY = 'mopa-heightmap.studio-state';
export const STUDIO_HISTORY_LIMIT = 20;
export const BLOB_CACHE_MAX_BYTES = 200 * 1024 * 1024;
export const DEFAULT_MASK_BACKEND = 'threshold';
export const DEFAULT_CLICKER_KEY = 'flood-fill';
export const DEFAULT_MASK_EDGE_SOFTNESS = 0;
export const DEFAULT_MASK_COVERAGE_PCT = 0;
export const DEFAULT_ACTIVE_ROUTE = 'wizard';
export const DEFAULT_WIZARD_PAGE = 0;
export const DEFAULT_EXPORT_PNG_ENABLED = true;
export const DEFAULT_EXPORT_LBRN2_ENABLED = true;
export const DEFAULT_EXPORT_STL_ENABLED = false;
export const LOCAL_STORAGE_DEBOUNCE_MS = 500;

export type ActiveRoute = 'wizard' | 'studio' | 'export';
export type MaskBackend = 'birefnet' | 'rembg' | 'threshold';
export type ClickerKey = 'flood-fill';
export interface SourceMeta {
  w: number;
  h: number;
  bytes: number;
}

export interface HistoryEntry {
  id: string;
  action: string;
  timestampIso: string;
  /**
   * Optional duration of the action in milliseconds. Populated when the
   * caller knows how long the underlying operation took (render / plan /
   * export); absent for instantaneous events like profile selection.
   */
  durationMs?: number;
}

export interface PassPlanEntry {
  passNumber: number;
  label: string;
  depthUm: number;
  colorHex: string;
}

export interface PassPlan {
  planId: string;
  passes: PassPlanEntry[];
}

export interface ToastMessage {
  id: string;
  severity: 'success' | 'info' | 'warn' | 'error';
  summary: string;
  detail: string;
}

export interface StudioState {
  session: {
    imageId: string | null;
    imageHash: string | null;
    sourceMeta: SourceMeta | null;
    history: HistoryEntry[];
  };
  pipeline: {
    mask: {
      backend: MaskBackend;
      clickerKey: ClickerKey;
      edgeSoftness: number;
      maskId: string | null;
      coveragePct: number;
    };
    render: {
      profileName: string | null;
    };
    /**
     * Mirrors the backend ``HeightmapSettings`` schema. Defaults match
     * ``DEFAULT_HEIGHTMAP_SETTINGS`` (which mirror the Pydantic defaults
     * in ``apps/api/schemas.py``). The render service forwards this
     * object verbatim to ``POST /render``.
     */
    settings: HeightmapSettings;
  };
  output: {
    heightmapId: string | null;
    previewId: string | null;
    plan: PassPlan | null;
    elapsedSeconds: number | null;
  };
  ui: {
    activeRoute: ActiveRoute;
    wizardPage: 0 | 1 | 2 | 3 | 4;
    rightPaneCollapsed: boolean;
    exportPngEnabled: boolean;
    exportLbrn2Enabled: boolean;
    exportStlEnabled: boolean;
    toasts: ToastMessage[];
  };
}

/**
 * Backend-default values for ``HeightmapSettings``. Keep in sync with
 * ``apps/api/schemas.py`` — the openapi drift-guard catches divergence
 * but the dev-time render request uses these directly.
 */
export const DEFAULT_HEIGHTMAP_SETTINGS: Required<HeightmapSettings> = {
  // Pre-sculptok input prep — all default-off.
  input_white_balance: false,
  input_clahe: false,
  input_clahe_clip: 2.0,
  input_clahe_grid: 8,
  input_denoise: false,
  input_denoise_strength: 5.0,
  input_remove_specular: false,
  input_specular_threshold: 245,
  input_max_dim: 0,
  input_auto_orient_face: false,
  input_auto_crop: false,
  input_auto_crop_aspect: 0.0,
  input_auto_crop_prefer_face: true,

  // External heightmap source. Set by upload / sculptok auto-pull,
  // not directly editable in the UI.
  external_heightmap_path: '',
  external_heightmap_polarity: 'bright_raised',

  // Polarity invert for signet rings / recessed designs.
  polarity_invert: false,

  // Subject mask deliverable (separate artifact, not applied to heightmap).
  subject_mask_enabled: false,
  subject_mask_backend: 'rembg',
  subject_mask_feather_px: 3,
  subject_mask_threshold: 0.5,

  // Procedural background pattern (composited onto the photo's
  // background pixels before the heightmap is loaded).
  background_pattern: 'none',
  background_scale: 1.0,
  background_angle: 0.0,
  background_seed: 0,
  background_intensity: 0.6,

  // LightBurn 3D Sliced polarity.
  black_is_deep: true,
  background_value: 1.0,

  // Heightmap output dither.
  dither: false,
  dither_levels: 256,

  // Refinement passes — opt-in.
  pre_clean_enabled: false,
  photo_tonal_enabled: false,
  photo_tonal_invert: false,
  photo_tonal_dither: true,
  photo_tonal_levels: 32,
  photo_tonal_strength: 0.7,
  photo_tonal_depth_fraction: 0.4,

  // Signature pass.
  signature_text: '',
  signature_corner: 'br',
  signature_height_fraction: 0.04,
  signature_margin_fraction: 0.03,
  signature_depth_fraction: 0.6,
};

export const DEFAULT_STUDIO_STATE: StudioState = {
  session: {
    imageId: null,
    imageHash: null,
    sourceMeta: null,
    history: [],
  },
  pipeline: {
    mask: {
      backend: DEFAULT_MASK_BACKEND,
      clickerKey: DEFAULT_CLICKER_KEY,
      edgeSoftness: DEFAULT_MASK_EDGE_SOFTNESS,
      maskId: null,
      coveragePct: DEFAULT_MASK_COVERAGE_PCT,
    },
    render: {
      profileName: null,
    },
    settings: { ...DEFAULT_HEIGHTMAP_SETTINGS },
  },
  output: {
    heightmapId: null,
    previewId: null,
    plan: null,
    elapsedSeconds: null,
  },
  ui: {
    activeRoute: DEFAULT_ACTIVE_ROUTE,
    wizardPage: DEFAULT_WIZARD_PAGE,
    rightPaneCollapsed: false,
    exportPngEnabled: DEFAULT_EXPORT_PNG_ENABLED,
    exportLbrn2Enabled: DEFAULT_EXPORT_LBRN2_ENABLED,
    exportStlEnabled: DEFAULT_EXPORT_STL_ENABLED,
    toasts: [],
  },
};

export function cloneDefaultStudioState(): StudioState {
  return structuredClone(DEFAULT_STUDIO_STATE);
}

export function serializeStudioState(state: StudioState): string {
  return JSON.stringify(state);
}

export function deserializeStudioState(raw: string | null): StudioState {
  if (!raw) {
    return cloneDefaultStudioState();
  }

  try {
    const parsed = JSON.parse(raw) as Partial<StudioState>;
    return {
      ...cloneDefaultStudioState(),
      ...parsed,
      session: { ...cloneDefaultStudioState().session, ...parsed.session },
      pipeline: {
        ...cloneDefaultStudioState().pipeline,
        ...parsed.pipeline,
        mask: { ...cloneDefaultStudioState().pipeline.mask, ...parsed.pipeline?.mask },
        render: { ...cloneDefaultStudioState().pipeline.render, ...parsed.pipeline?.render },
        settings: {
          ...cloneDefaultStudioState().pipeline.settings,
          ...parsed.pipeline?.settings,
        },
      },
      output: { ...cloneDefaultStudioState().output, ...parsed.output },
      ui: { ...cloneDefaultStudioState().ui, ...parsed.ui },
    };
  } catch {
    return cloneDefaultStudioState();
  }
}