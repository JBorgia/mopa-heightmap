/**
 * Unit tests for MaskService.
 *
 * Critical BUG-2 regression test: clickRefine() MUST send `clicker_key` from
 * the ClickerKey state — never the `backend` (MaskBackend) value.
 */
import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { describe, it, expect, vi, beforeEach } from 'vitest';

import { ClickMaskRequest, MaskResponse } from '../api/api-types';
import { ApiClientService } from '../api/api-client.service';
import { MaskService } from './mask.service';
import { SessionTreeService } from './session-tree.service';
import {
  DEFAULT_STUDIO_STATE,
  DEFAULT_CLICKER_KEY,
} from './studio-state';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const MASK_RESPONSE: MaskResponse = { mask_id: 'mask-111', coverage_pct: 42.5 };

function makeTreeMock(stateOverrides: Record<string, unknown> = {}) {
  const state = {
    ...DEFAULT_STUDIO_STATE,
    session: {
      ...DEFAULT_STUDIO_STATE.session,
      imageId: 'img-001',
    },
    pipeline: {
      ...DEFAULT_STUDIO_STATE.pipeline,
      mask: {
        ...DEFAULT_STUDIO_STATE.pipeline.mask,
        backend: 'rembg' as const,
        clickerKey: DEFAULT_CLICKER_KEY,
      },
    },
    ...stateOverrides,
  };

  return {
    state: vi.fn(() => state),
    patchState: vi.fn(),
    pushHistory: vi.fn(),
    _state: state,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('MaskService', () => {
  let service: MaskService;
  let apiMock: {
    createMask: ReturnType<typeof vi.fn>;
    clickMask: ReturnType<typeof vi.fn>;
  };
  let treeMock: ReturnType<typeof makeTreeMock>;

  beforeEach(() => {
    apiMock = {
      createMask: vi.fn(() => of(MASK_RESPONSE)),
      clickMask: vi.fn(() => of(MASK_RESPONSE)),
    };
    treeMock = makeTreeMock();

    TestBed.configureTestingModule({
      providers: [
        MaskService,
        { provide: ApiClientService, useValue: apiMock },
        { provide: SessionTreeService, useValue: treeMock },
      ],
    });

    service = TestBed.inject(MaskService);
  });

  // ── setBackend ───────────────────────────────────────────────────────

  it('setBackend patches pipeline.mask.backend', () => {
    service.setBackend('birefnet');
    expect(treeMock.patchState).toHaveBeenCalledOnce();
    const patcher = treeMock.patchState.mock.calls[0][0];
    const next = patcher(treeMock._state);
    expect(next.pipeline.mask.backend).toBe('birefnet');
  });

  // ── setClickerKey ────────────────────────────────────────────────────

  it('setClickerKey patches pipeline.mask.clickerKey (not backend)', () => {
    service.setClickerKey('flood-fill');
    const patcher = treeMock.patchState.mock.calls[0][0];
    const next = patcher(treeMock._state);
    expect(next.pipeline.mask.clickerKey).toBe('flood-fill');
    // backend must NOT be touched
    expect(next.pipeline.mask.backend).toBe('rembg');
  });

  // ── setEdgeSoftness ──────────────────────────────────────────────────

  it('setEdgeSoftness patches pipeline.mask.edgeSoftness', () => {
    service.setEdgeSoftness(0.7);
    const patcher = treeMock.patchState.mock.calls[0][0];
    const next = patcher(treeMock._state);
    expect(next.pipeline.mask.edgeSoftness).toBeCloseTo(0.7);
  });

  // ── createMask ───────────────────────────────────────────────────────

  it('createMask does nothing if imageId is null', () => {
    treeMock.state.mockReturnValue({
      ...treeMock._state,
      session: { ...DEFAULT_STUDIO_STATE.session, imageId: null as unknown as string },
    });
    service.createMask();
    expect(apiMock.createMask).not.toHaveBeenCalled();
  });

  it('createMask calls apiClient.createMask with backend and edge_softness', () => {
    service.createMask();
    expect(apiMock.createMask).toHaveBeenCalledWith({
      image_id: 'img-001',
      backend: 'rembg',
      edge_softness: DEFAULT_STUDIO_STATE.pipeline.mask.edgeSoftness,
    });
  });

  it('createMask patches mask state on success', () => {
    service.createMask();
    const patcher = treeMock.patchState.mock.calls[0][0];
    const next = patcher(treeMock._state);
    expect(next.pipeline.mask.maskId).toBe('mask-111');
    expect(next.pipeline.mask.coveragePct).toBeCloseTo(42.5);
  });

  it('createMask pushes history mask:create', () => {
    service.createMask();
    expect(treeMock.pushHistory).toHaveBeenCalledWith('mask:create');
  });

  // ── clickRefine — BUG-2 regression ──────────────────────────────────

  it('clickRefine does nothing if imageId is null', () => {
    treeMock.state.mockReturnValue({
      ...treeMock._state,
      session: { ...DEFAULT_STUDIO_STATE.session, imageId: null as unknown as string },
    });
    service.clickRefine(100, 200, 'positive');
    expect(apiMock.clickMask).not.toHaveBeenCalled();
  });

  it('BUG-2: clickRefine sends clicker_key, NOT backend', () => {
    service.clickRefine(100, 200, 'positive');
    const req: ClickMaskRequest = apiMock.clickMask.mock.calls[0][0];
    // clicker_key must come from ClickerKey state ('flood-fill'), not MaskBackend ('rembg')
    expect(req.clicker_key).toBe(DEFAULT_CLICKER_KEY);
    expect(req.clicker_key).not.toBe('rembg');
    expect(req.clicker_key).not.toBe('birefnet');
  });

  it('clickRefine sends correct image_id, x, y, label', () => {
    service.clickRefine(320, 240, 'negative');
    const req: ClickMaskRequest = apiMock.clickMask.mock.calls[0][0];
    expect(req.image_id).toBe('img-001');
    expect(req.x).toBe(320);
    expect(req.y).toBe(240);
    expect(req.label).toBe('negative');
  });

  it('clickRefine patches maskId and coveragePct on success', () => {
    service.clickRefine(10, 10, 'positive');
    const patcher = treeMock.patchState.mock.calls[0][0];
    const next = patcher(treeMock._state);
    expect(next.pipeline.mask.maskId).toBe('mask-111');
    expect(next.pipeline.mask.coveragePct).toBeCloseTo(42.5);
  });

  it('clickRefine pushes history with label', () => {
    service.clickRefine(10, 10, 'positive');
    expect(treeMock.pushHistory).toHaveBeenCalledWith('mask:click-refine:positive');
  });

  it('clickRefine negative label pushes correct history', () => {
    service.clickRefine(10, 10, 'negative');
    expect(treeMock.pushHistory).toHaveBeenCalledWith('mask:click-refine:negative');
  });
});
