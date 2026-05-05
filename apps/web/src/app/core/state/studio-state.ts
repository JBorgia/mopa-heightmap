export const STUDIO_STATE_STORAGE_KEY = 'mopa-heightmap.studio-state';
export const STUDIO_HISTORY_LIMIT = 20;
export const BLOB_CACHE_MAX_BYTES = 200 * 1024 * 1024;
export const DEFAULT_MASK_BACKEND = 'threshold';
export const DEFAULT_CLICKER_KEY = 'flood-fill';
export const DEFAULT_MASK_EDGE_SOFTNESS = 0;
export const DEFAULT_MASK_COVERAGE_PCT = 0;
export const DEFAULT_DETAIL_BALANCE = 0.35;
export const DEFAULT_RELIEF = 1;
export const DEFAULT_TARGET_MP = 2;
export const DEFAULT_SHARPEN = 0.2;
export const DEFAULT_BILATERAL_STRENGTH = 0.08;
export const DEFAULT_ACTIVE_ROUTE = 'wizard';
export const DEFAULT_WIZARD_PAGE = 0;
export const DEFAULT_EXPORT_PNG_ENABLED = true;
export const DEFAULT_EXPORT_LBRN2_ENABLED = true;
export const DEFAULT_EXPORT_STL_ENABLED = false;
export const LOCAL_STORAGE_DEBOUNCE_MS = 500;

export type ActiveRoute = 'wizard' | 'studio' | 'export';
export type MaskBackend = 'birefnet' | 'rembg' | 'threshold';
export type ClickerKey = 'flood-fill';
export type UpscalerKey = 'realesrgan' | 'swinir';

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

export interface ToneCurvePoint {
  input: number;
  output: number;
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
      detailBalance: number;
      multires: boolean;
      relief: number;
      profileName: string | null;
    };
    advanced: {
      preUpscale: boolean;
      upscaler: UpscalerKey;
      targetMP: number;
      sharpen: number;
      toneCurve: ToneCurvePoint[];
      bilateralStrength: number;
    };
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
      detailBalance: DEFAULT_DETAIL_BALANCE,
      multires: false,
      relief: DEFAULT_RELIEF,
      profileName: null,
    },
    advanced: {
      preUpscale: false,
      upscaler: 'realesrgan',
      targetMP: DEFAULT_TARGET_MP,
      sharpen: DEFAULT_SHARPEN,
      toneCurve: [
        { input: 0, output: 0 },
        { input: 1, output: 1 },
      ],
      bilateralStrength: DEFAULT_BILATERAL_STRENGTH,
    },
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
        advanced: { ...cloneDefaultStudioState().pipeline.advanced, ...parsed.pipeline?.advanced },
      },
      output: { ...cloneDefaultStudioState().output, ...parsed.output },
      ui: { ...cloneDefaultStudioState().ui, ...parsed.ui },
    };
  } catch {
    return cloneDefaultStudioState();
  }
}