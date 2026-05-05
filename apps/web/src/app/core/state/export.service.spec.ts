/**
 * Unit tests for ExportService — verifies guards, API delegation, and
 * the browser download trigger. All HTTP calls and state reads are mocked.
 */
import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { describe, it, expect, vi, beforeEach } from 'vitest';

import { ApiClientService } from '../api/api-client.service';
import {
  EXPORT_PNG_FILENAME,
  EXPORT_LBRN2_FILENAME,
  EXPORT_STL_FILENAME,
  EXPORT_STL_DEFAULT_Z_SCALE_MM,
  EXPORT_STL_DEFAULT_BASE_THICKNESS_MM,
  ExportService,
} from './export.service';
import { SessionTreeService } from './session-tree.service';
import { DEFAULT_STUDIO_STATE } from './studio-state';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeState(overrides: Record<string, unknown> = {}) {
  return {
    ...DEFAULT_STUDIO_STATE,
    output: {
      ...DEFAULT_STUDIO_STATE.output,
      ...overrides,
    },
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('ExportService', () => {
  let service: ExportService;
  let apiClient: ReturnType<typeof vi.fn>;
  let sessionTree: ReturnType<typeof vi.fn>;
  let mockState: ReturnType<typeof makeState>;
  let triggerSpy: ReturnType<typeof vi.spyOn> | null = null;

  beforeEach(() => {
    mockState = makeState();

    const apiMock = {
      exportPng: vi.fn(() => of(new Blob(['png'], { type: 'image/png' }))),
      exportLbrn2: vi.fn(() => of(new Blob(['xml'], { type: 'application/xml' }))),
      exportStl: vi.fn(() => of(new Blob(['stl'], { type: 'model/stl' }))),
    };

    const treeMock = {
      state: vi.fn(() => mockState),
      output: vi.fn(() => mockState.output),
      pushHistory: vi.fn(),
    };

    TestBed.configureTestingModule({
      providers: [
        ExportService,
        { provide: ApiClientService, useValue: apiMock },
        { provide: SessionTreeService, useValue: treeMock },
      ],
    });

    service = TestBed.inject(ExportService);
    apiClient = TestBed.inject(ApiClientService) as unknown as ReturnType<typeof vi.fn>;
    sessionTree = TestBed.inject(SessionTreeService) as unknown as ReturnType<typeof vi.fn>;

    // Prevent actual DOM link clicks in tests.
    triggerSpy = vi.spyOn(service as unknown as { _triggerDownload: () => void }, '_triggerDownload').mockImplementation(() => undefined);
  });

  // ── constants ────────────────────────────────────────────────────────────

  it('exports EXPORT_PNG_FILENAME constant', () => {
    expect(EXPORT_PNG_FILENAME).toBe('heightmap.png');
  });

  it('exports EXPORT_LBRN2_FILENAME constant', () => {
    expect(EXPORT_LBRN2_FILENAME).toBe('project.lbrn2');
  });

  it('exports EXPORT_STL_FILENAME constant', () => {
    expect(EXPORT_STL_FILENAME).toBe('heightmap.stl');
  });

  it('exports correct STL default parameters', () => {
    expect(EXPORT_STL_DEFAULT_Z_SCALE_MM).toBe(5.0);
    expect(EXPORT_STL_DEFAULT_BASE_THICKNESS_MM).toBe(2.0);
  });

  // ── exportPng ─────────────────────────────────────────────────────────

  it('exportPng does nothing when heightmapId is null', () => {
    mockState = makeState({ heightmapId: null });
    (sessionTree as unknown as { state: ReturnType<typeof vi.fn> }).state.mockReturnValue(mockState);
    service.exportPng();
    expect((apiClient as unknown as { exportPng: ReturnType<typeof vi.fn> }).exportPng).not.toHaveBeenCalled();
  });

  it('exportPng calls apiClient.exportPng with heightmap_id and bit_depth 16', () => {
    mockState = makeState({ heightmapId: 'hm-abc' });
    (sessionTree as unknown as { state: ReturnType<typeof vi.fn> }).state.mockReturnValue(mockState);
    service.exportPng();
    expect((apiClient as unknown as { exportPng: ReturnType<typeof vi.fn> }).exportPng).toHaveBeenCalledWith({
      heightmap_id: 'hm-abc',
      bit_depth: 16,
    });
  });

  it('exportPng pushes history after success', () => {
    mockState = makeState({ heightmapId: 'hm-abc' });
    (sessionTree as unknown as { state: ReturnType<typeof vi.fn> }).state.mockReturnValue(mockState);
    service.exportPng();
    expect((sessionTree as unknown as { pushHistory: ReturnType<typeof vi.fn> }).pushHistory).toHaveBeenCalledWith('export:png');
  });

  // ── exportLbrn2 ───────────────────────────────────────────────────────

  it('exportLbrn2 does nothing when plan is null', () => {
    mockState = makeState({ plan: null, heightmapId: 'hm-abc' });
    (sessionTree as unknown as { state: ReturnType<typeof vi.fn> }).state.mockReturnValue(mockState);
    service.exportLbrn2();
    expect((apiClient as unknown as { exportLbrn2: ReturnType<typeof vi.fn> }).exportLbrn2).not.toHaveBeenCalled();
  });

  it('exportLbrn2 does nothing when heightmapId is null', () => {
    mockState = makeState({
      plan: { planId: 'plan-1', passes: [] },
      heightmapId: null,
    });
    (sessionTree as unknown as { state: ReturnType<typeof vi.fn> }).state.mockReturnValue(mockState);
    service.exportLbrn2();
    expect((apiClient as unknown as { exportLbrn2: ReturnType<typeof vi.fn> }).exportLbrn2).not.toHaveBeenCalled();
  });

  it('exportLbrn2 calls apiClient with plan_id and heightmap_id', () => {
    mockState = makeState({
      plan: { planId: 'plan-1', passes: [] },
      heightmapId: 'hm-abc',
    });
    (sessionTree as unknown as { state: ReturnType<typeof vi.fn> }).state.mockReturnValue(mockState);
    service.exportLbrn2();
    expect((apiClient as unknown as { exportLbrn2: ReturnType<typeof vi.fn> }).exportLbrn2).toHaveBeenCalledWith(
      expect.objectContaining({ plan_id: 'plan-1', heightmap_id: 'hm-abc' }),
    );
  });

  it('exportLbrn2 pushes history export:lbrn2', () => {
    mockState = makeState({
      plan: { planId: 'plan-2', passes: [] },
      heightmapId: 'hm-xyz',
    });
    (sessionTree as unknown as { state: ReturnType<typeof vi.fn> }).state.mockReturnValue(mockState);
    service.exportLbrn2();
    expect((sessionTree as unknown as { pushHistory: ReturnType<typeof vi.fn> }).pushHistory).toHaveBeenCalledWith('export:lbrn2');
  });

  // ── exportStl ─────────────────────────────────────────────────────────

  it('exportStl does nothing when heightmapId is null', () => {
    mockState = makeState({ heightmapId: null });
    (sessionTree as unknown as { state: ReturnType<typeof vi.fn> }).state.mockReturnValue(mockState);
    service.exportStl();
    expect((apiClient as unknown as { exportStl: ReturnType<typeof vi.fn> }).exportStl).not.toHaveBeenCalled();
  });

  it('exportStl calls apiClient with correct defaults', () => {
    mockState = makeState({ heightmapId: 'hm-abc' });
    (sessionTree as unknown as { state: ReturnType<typeof vi.fn> }).state.mockReturnValue(mockState);
    service.exportStl();
    expect((apiClient as unknown as { exportStl: ReturnType<typeof vi.fn> }).exportStl).toHaveBeenCalledWith({
      heightmap_id: 'hm-abc',
      z_scale_mm: EXPORT_STL_DEFAULT_Z_SCALE_MM,
      base_thickness_mm: EXPORT_STL_DEFAULT_BASE_THICKNESS_MM,
    });
  });

  it('exportStl pushes history export:stl', () => {
    mockState = makeState({ heightmapId: 'hm-abc' });
    (sessionTree as unknown as { state: ReturnType<typeof vi.fn> }).state.mockReturnValue(mockState);
    service.exportStl();
    expect((sessionTree as unknown as { pushHistory: ReturnType<typeof vi.fn> }).pushHistory).toHaveBeenCalledWith('export:stl');
  });
});
