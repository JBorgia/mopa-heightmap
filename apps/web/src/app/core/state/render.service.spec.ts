import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { describe, it, expect, vi, beforeEach } from 'vitest';

import { ApiClientService } from '../api/api-client.service';
import { BlobCacheService } from './blob-cache.service';
import { RenderService } from './render.service';
import { SessionTreeService } from './session-tree.service';
import { DEFAULT_STUDIO_STATE } from './studio-state';

const RENDER_RESPONSE = {
  image_hash: 'hash-abc',
  heightmap_id: 'hm-123',
  preview_id: 'pv-456',
  elapsed_s: 1.23,
};

function makeTreeMock() {
  const state = {
    ...DEFAULT_STUDIO_STATE,
    session: { ...DEFAULT_STUDIO_STATE.session, imageId: 'img-001' },
  };
  return {
    state: vi.fn(() => state),
    patchState: vi.fn(),
    pushHistory: vi.fn(),
    _state: state,
  };
}

describe('RenderService', () => {
  let service: RenderService;
  let treeMock: ReturnType<typeof makeTreeMock>;
  let apiMock: { render: ReturnType<typeof vi.fn> };
  let blobMock: { get: ReturnType<typeof vi.fn> };

  beforeEach(() => {
    treeMock = makeTreeMock();
    apiMock = { render: vi.fn(() => of(RENDER_RESPONSE)) };
    blobMock = { get: vi.fn(() => null) };

    TestBed.configureTestingModule({
      providers: [
        RenderService,
        { provide: SessionTreeService, useValue: treeMock },
        { provide: ApiClientService, useValue: apiMock },
        { provide: BlobCacheService, useValue: blobMock },
      ],
    });
    service = TestBed.inject(RenderService);
  });

  // --- setDetailBalance -------------------------------------------------------

  it('setDetailBalance patches pipeline.render.detailBalance', () => {
    service.setDetailBalance(0.75);
    expect(treeMock.patchState).toHaveBeenCalledOnce();
    const updater = treeMock.patchState.mock.calls[0][0];
    const result = updater(treeMock._state);
    expect(result.pipeline.render.detailBalance).toBe(0.75);
  });

  // --- setMultires ------------------------------------------------------------

  it('setMultires patches pipeline.render.multires', () => {
    service.setMultires(true);
    expect(treeMock.patchState).toHaveBeenCalledOnce();
    const updater = treeMock.patchState.mock.calls[0][0];
    const result = updater(treeMock._state);
    expect(result.pipeline.render.multires).toBe(true);
  });

  // --- setRelief --------------------------------------------------------------

  it('setRelief patches pipeline.render.relief', () => {
    service.setRelief(2.5);
    expect(treeMock.patchState).toHaveBeenCalledOnce();
    const updater = treeMock.patchState.mock.calls[0][0];
    const result = updater(treeMock._state);
    expect(result.pipeline.render.relief).toBe(2.5);
  });

  // --- render -----------------------------------------------------------------

  it('render does nothing if imageId is null', () => {
    treeMock.state.mockReturnValue({
      ...treeMock._state,
      session: { ...DEFAULT_STUDIO_STATE.session, imageId: null as unknown as string },
    });
    service.render();
    expect(apiMock.render).not.toHaveBeenCalled();
  });

  it('render calls apiClient.render with image_id and profile_name', () => {
    service.render();
    expect(apiMock.render).toHaveBeenCalledWith({
      image_id: 'img-001',
      profile_name: DEFAULT_STUDIO_STATE.pipeline.render.profileName,
    });
  });

  it('render patches output and session after success', () => {
    service.render();
    expect(treeMock.patchState).toHaveBeenCalledOnce();
    const updater = treeMock.patchState.mock.calls[0][0];
    const result = updater(treeMock._state);
    expect(result.session.imageHash).toBe('hash-abc');
    expect(result.output.heightmapId).toBe('hm-123');
    expect(result.output.previewId).toBe('pv-456');
    expect(result.output.elapsedSeconds).toBe(1.23);
  });

  it('render probes blobCache after success', () => {
    service.render();
    expect(blobMock.get).toHaveBeenCalledWith('hash-abc');
  });

  it('render pushes render:run history', () => {
    service.render();
    expect(treeMock.pushHistory).toHaveBeenCalledWith('render:run');
  });
});
