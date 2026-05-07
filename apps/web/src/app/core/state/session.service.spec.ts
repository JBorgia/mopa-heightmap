import { TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';
import { describe, it, expect, vi, beforeEach } from 'vitest';

import { ApiClientService } from '../api/api-client.service';
import { SessionService } from './session.service';
import { SessionTreeService } from './session-tree.service';
import { DEFAULT_STUDIO_STATE } from './studio-state';

const PROFILES = [{ name: 'default', display_name: 'Default' }];
const UPLOAD_RESPONSE = { image_id: 'img-new', sha256: 'abc123', w: 800, h: 600 };

function makeTreeMock() {
  const state = { ...DEFAULT_STUDIO_STATE };
  return {
    state: vi.fn(() => state),
    pipeline: vi.fn(() => state.pipeline),
    patchState: vi.fn(),
    pushHistory: vi.fn(),
    addToast: vi.fn(),
    _state: state,
  };
}

describe('SessionService', () => {
  let service: SessionService;
  let treeMock: ReturnType<typeof makeTreeMock>;
  let apiMock: {
    listProfiles: ReturnType<typeof vi.fn>;
    uploadImage: ReturnType<typeof vi.fn>;
  };

  beforeEach(() => {
    treeMock = makeTreeMock();
    apiMock = {
      listProfiles: vi.fn(() => of(PROFILES)),
      uploadImage: vi.fn(() => of(UPLOAD_RESPONSE)),
    };

    TestBed.configureTestingModule({
      providers: [
        SessionService,
        { provide: SessionTreeService, useValue: treeMock },
        { provide: ApiClientService, useValue: apiMock },
      ],
    });
    service = TestBed.inject(SessionService);
  });

  // --- initial signals -------------------------------------------------------

  it('profilesLoaded starts false', () => {
    expect(service.profilesLoaded()).toBe(false);
  });

  it('uploadInFlight starts false', () => {
    expect(service.uploadInFlight()).toBe(false);
  });

  // --- loadProfiles -----------------------------------------------------------

  it('loadProfiles() calls apiClient.listProfiles and sets profiles', () => {
    service.loadProfiles();
    expect(apiMock.listProfiles).toHaveBeenCalledOnce();
    expect(service.profiles()).toEqual(PROFILES);
    expect(service.profilesLoaded()).toBe(true);
  });

  it('loadProfiles() does not call API if already loaded', () => {
    service.loadProfiles();
    service.loadProfiles();
    expect(apiMock.listProfiles).toHaveBeenCalledOnce();
  });

  it('loadProfiles() auto-selects first profile when none is set', () => {
    // pipeline.render.profileName is null by default
    treeMock.pipeline.mockReturnValue({
      ...DEFAULT_STUDIO_STATE.pipeline,
      render: { ...DEFAULT_STUDIO_STATE.pipeline.render, profileName: null },
    });
    service.loadProfiles();
    expect(treeMock.patchState).toHaveBeenCalled();
  });

  it('loadProfiles() does NOT override existing profile selection', () => {
    treeMock.pipeline.mockReturnValue({
      ...DEFAULT_STUDIO_STATE.pipeline,
      render: { ...DEFAULT_STUDIO_STATE.pipeline.render, profileName: 'custom' },
    });
    service.loadProfiles();
    expect(treeMock.patchState).not.toHaveBeenCalled();
  });

  // --- uploadImage ------------------------------------------------------------

  it('uploadImage() sets uploadInFlight=true before response', () => {
    apiMock.uploadImage.mockReturnValue({ subscribe: vi.fn() }); // never resolves
    service.uploadImage(new File(['x'], 'test.png'));
    expect(service.uploadInFlight()).toBe(true);
  });

  it('uploadImage() patches session state on success', () => {
    const file = new File(['x'], 'photo.jpg');
    Object.defineProperty(file, 'size', { value: 12345 });
    service.uploadImage(file);
    expect(treeMock.patchState).toHaveBeenCalled();
    const updater = treeMock.patchState.mock.calls[0][0];
    const result = updater(treeMock._state);
    expect(result.session.imageId).toBe('img-new');
    expect(result.session.imageHash).toBe('abc123');
    expect(result.session.sourceMeta).toEqual({ w: 800, h: 600, bytes: 12345 });
    expect(result.pipeline.mask.maskId).toBeNull();
    expect(result.output.heightmapId).toBeNull();
  });

  it('uploadImage() pushes history on success', () => {
    const file = new File(['x'], 'photo.png');
    service.uploadImage(file);
    expect(treeMock.pushHistory).toHaveBeenCalledWith('session:upload:photo.png');
  });

  it('uploadImage() resets uploadInFlight on complete', () => {
    service.uploadImage(new File(['x'], 'test.png'));
    // of() auto-completes
    expect(service.uploadInFlight()).toBe(false);
  });

  it('uploadImage() resets uploadInFlight on error', () => {
    apiMock.uploadImage.mockReturnValue(throwError(() => new Error('network')));
    service.uploadImage(new File(['x'], 'bad.png'));
    expect(service.uploadInFlight()).toBe(false);
  });

  it('uploadImage() surfaces an error toast on failure (silently swallowing it left users guessing)', () => {
    apiMock.uploadImage.mockReturnValue(throwError(() => ({ error: { detail: 'Cannot decode image' } })));
    service.uploadImage(new File(['x'], 'bad.png'));
    expect(treeMock.addToast).toHaveBeenCalledWith(expect.objectContaining({
      severity: 'error',
      summary: 'Upload failed',
      detail: 'Cannot decode image',
    }));
  });

  it('uploadImage() clears the previous heightmap source — carrying it over would silently render the wrong depth', () => {
    const file = new File(['x'], 'new.jpg');
    service.uploadImage(file);
    const updater = treeMock.patchState.mock.calls[0][0];
    const stateWithHeightmap = {
      ...DEFAULT_STUDIO_STATE,
      pipeline: {
        ...DEFAULT_STUDIO_STATE.pipeline,
        settings: {
          ...DEFAULT_STUDIO_STATE.pipeline.settings,
          external_heightmap_path: '/tmp/old-subject.png',
        },
      },
    };
    const result = updater(stateWithHeightmap);
    expect(result.pipeline.settings.external_heightmap_path).toBe('');
  });

  // --- setProfileName ---------------------------------------------------------

  it('setProfileName() patches pipeline.render.profileName', () => {
    service.setProfileName('standard');
    expect(treeMock.patchState).toHaveBeenCalled();
    const updater = treeMock.patchState.mock.calls[0][0];
    const result = updater(treeMock._state);
    expect(result.pipeline.render.profileName).toBe('standard');
  });

  it('setProfileName(null) patches to null', () => {
    service.setProfileName(null);
    const updater = treeMock.patchState.mock.calls[0][0];
    const result = updater(treeMock._state);
    expect(result.pipeline.render.profileName).toBeNull();
  });

  it('setProfileName() pushes profile:select history', () => {
    service.setProfileName('fast');
    expect(treeMock.pushHistory).toHaveBeenCalledWith('profile:select:fast');
  });

  it('setProfileName(null) pushes profile:select:none history', () => {
    service.setProfileName(null);
    expect(treeMock.pushHistory).toHaveBeenCalledWith('profile:select:none');
  });

  it('setProfileName() clears the existing pass plan when the profile actually changes', () => {
    service.setProfileName('mopa_60w_steel');
    const updater = treeMock.patchState.mock.calls[0][0];
    const stateWithPlan = {
      ...DEFAULT_STUDIO_STATE,
      pipeline: {
        ...DEFAULT_STUDIO_STATE.pipeline,
        render: { ...DEFAULT_STUDIO_STATE.pipeline.render, profileName: 'mopa_60w_brass' },
      },
      output: {
        ...DEFAULT_STUDIO_STATE.output,
        plan: { planId: 'p-old', passes: [] },
      },
    };
    const result = updater(stateWithPlan);
    expect(result.pipeline.render.profileName).toBe('mopa_60w_steel');
    expect(result.output.plan).toBeNull();
  });

  it('setProfileName() leaves the plan alone when the profile did not change', () => {
    service.setProfileName('mopa_60w_brass');
    const updater = treeMock.patchState.mock.calls[0][0];
    const stateWithPlan = {
      ...DEFAULT_STUDIO_STATE,
      pipeline: {
        ...DEFAULT_STUDIO_STATE.pipeline,
        render: { ...DEFAULT_STUDIO_STATE.pipeline.render, profileName: 'mopa_60w_brass' },
      },
      output: {
        ...DEFAULT_STUDIO_STATE.output,
        plan: { planId: 'p-current', passes: [] },
      },
    };
    const result = updater(stateWithPlan);
    expect(result.output.plan?.planId).toBe('p-current');
  });
});
