/**
 * Tests for SculptokService.uploadHeightmap — verify the toast/state
 * branching when the server reports an auto-crop on a composite upload.
 */
import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { describe, it, expect, vi, beforeEach } from 'vitest';

import { ApiClientService } from '../api/api-client.service';
import { RenderService } from './render.service';
import { SculptokService } from './sculptok.service';
import { SessionTreeService } from './session-tree.service';

function makeRenderMock() {
  return { patchSettings: vi.fn() };
}

function makeTreeMock() {
  return {
    addToast: vi.fn(),
    pushHistory: vi.fn(),
  };
}

describe('SculptokService.uploadHeightmap', () => {
  let service: SculptokService;
  let apiMock: { uploadHeightmap: ReturnType<typeof vi.fn>; sculptokCredits: ReturnType<typeof vi.fn> };
  let renderMock: ReturnType<typeof makeRenderMock>;
  let treeMock: ReturnType<typeof makeTreeMock>;

  beforeEach(() => {
    apiMock = {
      uploadHeightmap: vi.fn(),
      sculptokCredits: vi.fn(() => of({ configured: false, balance: null })),
    };
    renderMock = makeRenderMock();
    treeMock = makeTreeMock();

    TestBed.configureTestingModule({
      providers: [
        SculptokService,
        { provide: ApiClientService, useValue: apiMock },
        { provide: RenderService, useValue: renderMock },
        { provide: SessionTreeService, useValue: treeMock },
      ],
    });
    service = TestBed.inject(SculptokService);
  });

  it('regular upload (auto_cropped=false) shows a success toast', () => {
    apiMock.uploadHeightmap.mockReturnValue(of({
      heightmap_path: '/tmp/out.png',
      width: 2048,
      height: 2048,
      auto_cropped: false,
    }));
    service.uploadHeightmap(new File(['x'], 'good.png'));
    expect(renderMock.patchSettings).toHaveBeenCalledWith('external_heightmap_path', '/tmp/out.png');
    const toast = treeMock.addToast.mock.calls[0][0];
    expect(toast.severity).toBe('success');
    expect(toast.summary).toBe('Heightmap uploaded');
  });

  it('composite upload (auto_cropped=true) shows a warn toast that explains the crop', () => {
    apiMock.uploadHeightmap.mockReturnValue(of({
      heightmap_path: '/tmp/out.png',
      width: 960,
      height: 1280,
      auto_cropped: true,
    }));
    service.uploadHeightmap(new File(['x'], 'composite.png'));
    expect(renderMock.patchSettings).toHaveBeenCalledWith('external_heightmap_path', '/tmp/out.png');
    const toast = treeMock.addToast.mock.calls[0][0];
    expect(toast.severity).toBe('warn');
    expect(toast.summary).toBe('Side-by-side composite detected');
    expect(toast.detail).toContain('960×1280');
    expect(toast.detail.toLowerCase()).toContain('depth-map');
  });
});
