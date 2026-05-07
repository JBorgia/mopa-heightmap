/**
 * End-to-end-style integration test for the wizard.
 *
 * Drives the full upload → mask → render → plan → export flow with mocked
 * HTTP, asserting each state transition. Catches regressions like:
 *   * stale-plan after re-render or profile change (caused .lbrn2 to be
 *     bundled with the wrong heightmap),
 *   * .lbrn2 download saved with the wrong extension,
 *   * page chips not reflecting completion state,
 *   * the auto-compute effect failing to fire when the user skips page 4.
 */
import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { of } from 'rxjs';
import { describe, it, expect, vi, beforeEach } from 'vitest';

import { ApiClientService } from '../../core/api/api-client.service';
import { SessionTreeService } from '../../core/state/session-tree.service';
import { WizardShellComponent } from './wizard-shell.component';

const UPLOAD_RESPONSE = {
  image_id: 'img-001',
  sha256: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
  w: 1920,
  h: 1280,
};

const MASK_RESPONSE = { mask_id: 'mask-001', coverage_pct: 42.5 };

const HEIGHTMAP_UPLOAD_RESPONSE = {
  heightmap_path: '/tmp/foo.png',
  width: 2048,
  height: 2048,
};

interface RenderResponseShape {
  heightmap_id: string;
  preview_id: string;
  conditioned_id: string | null;
  render_mask_id: string | null;
  elapsed_s: number;
  image_hash: string;
}

const RENDER_RESPONSE_V1: RenderResponseShape = {
  heightmap_id: 'hm-001',
  preview_id: 'preview-001',
  conditioned_id: 'cond-001',
  render_mask_id: null,
  elapsed_s: 1.23,
  image_hash: 'sha-aaa',
};

const RENDER_RESPONSE_V2: RenderResponseShape = {
  heightmap_id: 'hm-002', // new id after re-render with different settings
  preview_id: 'preview-002',
  conditioned_id: 'cond-002',
  render_mask_id: 'rmask-002',
  elapsed_s: 2.34,
  image_hash: 'sha-bbb',
};

const PLAN_RESPONSE_V1 = {
  plan_id: 'plan-001',
  passes: [
    { pass_number: 1, label: 'Depth', depth_um: 50, color_hex: '#ff0000' },
  ],
};

const PLAN_RESPONSE_V2 = {
  plan_id: 'plan-002',
  passes: [
    { pass_number: 1, label: 'Depth', depth_um: 50, color_hex: '#ff0000' },
    { pass_number: 2, label: 'Detail', depth_um: 20, color_hex: '#00ff00' },
  ],
};

function makeApiMock() {
  return {
    uploadImage: vi.fn(() => of(UPLOAD_RESPONSE)),
    uploadHeightmap: vi.fn(() => of(HEIGHTMAP_UPLOAD_RESPONSE)),
    listProfiles: vi.fn(() => of([{ name: 'mopa_60w_brass' }, { name: 'mopa_60w_steel' }])),
    createMask: vi.fn(() => of(MASK_RESPONSE)),
    clickMask: vi.fn(() => of(MASK_RESPONSE)),
    render: vi.fn(() => of(RENDER_RESPONSE_V1)),
    plan: vi.fn(() => of(PLAN_RESPONSE_V1)),
    exportPng: vi.fn(() => of(new Blob(['png'], { type: 'image/png' }))),
    exportLbrn2: vi.fn(() => of(new Blob(['zip'], { type: 'application/zip' }))),
    exportStl: vi.fn(() => of(new Blob(['stl'], { type: 'model/stl' }))),
    exportBundle: vi.fn(() => of(new Blob(['zip'], { type: 'application/zip' }))),
    sculptokCredits: vi.fn(() => of({ configured: false, balance: null })),
    sculptokGenerate: vi.fn(),
    blobUrl: (id: string) => `http://test/blob/${id}`,
  };
}

describe('WizardShellComponent — full flow', () => {
  let apiMock: ReturnType<typeof makeApiMock>;
  let sessionTree: SessionTreeService;

  beforeEach(async () => {
    apiMock = makeApiMock();
    await TestBed.configureTestingModule({
      imports: [WizardShellComponent],
      providers: [
        provideRouter([]),
        { provide: ApiClientService, useValue: apiMock },
      ],
    }).compileComponents();
    sessionTree = TestBed.inject(SessionTreeService);
    sessionTree.reset();
  });

  // ── Step 0: Upload ──────────────────────────────────────────────────────

  it('starts on page 0 with all chips incomplete', () => {
    const fixture = TestBed.createComponent(WizardShellComponent);
    fixture.detectChanges();
    expect(sessionTree.ui().wizardPage).toBe(0);
    const cmp = fixture.componentInstance as unknown as { pageStatus: (i: number) => string };
    expect(cmp.pageStatus(0)).toBe('incomplete');
    expect(cmp.pageStatus(2)).toBe('incomplete');
    expect(cmp.pageStatus(3)).toBe('incomplete');
  });

  it('upload populates session and marks step 1 complete', async () => {
    const fixture = TestBed.createComponent(WizardShellComponent);
    fixture.detectChanges();
    const cmp = fixture.componentInstance as unknown as {
      onFileSelected: (e: Event) => void;
      pageStatus: (i: number) => string;
    };
    const file = new File(['fake'], 'photo.jpg', { type: 'image/jpeg' });
    const event = { target: { files: { item: () => file }, value: '' } } as unknown as Event;
    cmp.onFileSelected(event);
    expect(apiMock.uploadImage).toHaveBeenCalledWith(file);
    expect(sessionTree.session().imageId).toBe(UPLOAD_RESPONSE.image_id);
    expect(sessionTree.session().sourceMeta?.w).toBe(1920);
    expect(cmp.pageStatus(0)).toBe('complete');
  });

  // ── Step 1: Mask (optional) ──────────────────────────────────────────────

  it('mask compute fills mask state and marks step 2 complete', () => {
    const fixture = TestBed.createComponent(WizardShellComponent);
    fixture.detectChanges();
    const cmp = fixture.componentInstance as unknown as {
      onFileSelected: (e: Event) => void;
      createMask: () => void;
      pageStatus: (i: number) => string;
    };
    cmp.onFileSelected({ target: { files: { item: () => new File(['x'], 'x.jpg') }, value: '' } } as unknown as Event);
    cmp.createMask();
    expect(apiMock.createMask).toHaveBeenCalled();
    expect(sessionTree.pipeline().mask.maskId).toBe('mask-001');
    expect(sessionTree.pipeline().mask.coveragePct).toBe(42.5);
    expect(cmp.pageStatus(1)).toBe('complete');
  });

  it('skipping mask leaves step 2 as optional-incomplete (status icon "·")', () => {
    const fixture = TestBed.createComponent(WizardShellComponent);
    fixture.detectChanges();
    const cmp = fixture.componentInstance as unknown as {
      pageStatusIcon: (i: number) => string;
      pageStatus: (i: number) => string;
    };
    expect(cmp.pageStatus(1)).toBe('incomplete');
    expect(cmp.pageStatusIcon(1)).toBe('·'); // optional, not done
    expect(cmp.pageStatusIcon(0)).toBe('○'); // required, not done
  });

  // ── Step 2: Heightmap source + render ────────────────────────────────────

  it('heightmap upload sets external_heightmap_path then render fills output', () => {
    const fixture = TestBed.createComponent(WizardShellComponent);
    fixture.detectChanges();
    const cmp = fixture.componentInstance as unknown as {
      onFileSelected: (e: Event) => void;
      onHeightmapFileSelected: (e: Event) => void;
      renderPreview: () => void;
      canRender: () => boolean;
      pageStatus: (i: number) => string;
    };
    cmp.onFileSelected({ target: { files: { item: () => new File(['x'], 'x.jpg') }, value: '' } } as unknown as Event);
    expect(cmp.canRender()).toBe(false); // no heightmap yet
    cmp.onHeightmapFileSelected({ target: { files: { item: () => new File(['x'], 'h.png') }, value: '' } } as unknown as Event);
    expect(sessionTree.pipeline().settings.external_heightmap_path).toBe('/tmp/foo.png');
    expect(cmp.canRender()).toBe(true);
    cmp.renderPreview();
    expect(apiMock.render).toHaveBeenCalled();
    expect(sessionTree.output().heightmapId).toBe('hm-001');
    expect(sessionTree.output().conditionedId).toBe('cond-001');
    expect(cmp.pageStatus(2)).toBe('complete');
  });

  it('render exposes conditioned + render-mask blob ids', () => {
    const fixture = TestBed.createComponent(WizardShellComponent);
    fixture.detectChanges();
    apiMock.render.mockReturnValue(of(RENDER_RESPONSE_V2));
    const cmp = fixture.componentInstance as unknown as {
      onFileSelected: (e: Event) => void;
      onHeightmapFileSelected: (e: Event) => void;
      renderPreview: () => void;
    };
    cmp.onFileSelected({ target: { files: { item: () => new File(['x'], 'x.jpg') }, value: '' } } as unknown as Event);
    cmp.onHeightmapFileSelected({ target: { files: { item: () => new File(['x'], 'h.png') }, value: '' } } as unknown as Event);
    cmp.renderPreview();
    expect(sessionTree.output().conditionedId).toBe('cond-002');
    expect(sessionTree.output().renderMaskId).toBe('rmask-002');
  });

  // ── Step 3: Auto-compute plan ────────────────────────────────────────────

  it('plan auto-computes when image + heightmap + profile are ready, regardless of current page', async () => {
    const fixture = TestBed.createComponent(WizardShellComponent);
    fixture.detectChanges(); // triggers loadProfiles → auto-selects first
    await fixture.whenStable();
    const cmp = fixture.componentInstance as unknown as {
      onFileSelected: (e: Event) => void;
      onHeightmapFileSelected: (e: Event) => void;
      renderPreview: () => void;
    };
    cmp.onFileSelected({ target: { files: { item: () => new File(['x'], 'x.jpg') }, value: '' } } as unknown as Event);
    cmp.onHeightmapFileSelected({ target: { files: { item: () => new File(['x'], 'h.png') }, value: '' } } as unknown as Event);
    cmp.renderPreview();
    // wizardPage is still 0 — but the auto-compute effect doesn't gate on page.
    fixture.detectChanges();
    await fixture.whenStable();
    expect(apiMock.plan).toHaveBeenCalled();
    expect(sessionTree.output().plan?.planId).toBe('plan-001');
  });

  // ── Bug regression: stale plan after re-render ───────────────────────────

  it('re-render clears the previous plan and triggers a fresh auto-compute', async () => {
    const fixture = TestBed.createComponent(WizardShellComponent);
    fixture.detectChanges();
    await fixture.whenStable();
    const cmp = fixture.componentInstance as unknown as {
      onFileSelected: (e: Event) => void;
      onHeightmapFileSelected: (e: Event) => void;
      renderPreview: () => void;
    };
    cmp.onFileSelected({ target: { files: { item: () => new File(['x'], 'x.jpg') }, value: '' } } as unknown as Event);
    cmp.onHeightmapFileSelected({ target: { files: { item: () => new File(['x'], 'h.png') }, value: '' } } as unknown as Event);
    cmp.renderPreview();
    fixture.detectChanges();
    await fixture.whenStable();
    expect(sessionTree.output().plan?.planId).toBe('plan-001');

    // New render returns a different heightmap id → plan must be invalidated.
    apiMock.render.mockReturnValue(of(RENDER_RESPONSE_V2));
    apiMock.plan.mockReturnValue(of(PLAN_RESPONSE_V2));
    cmp.renderPreview();
    fixture.detectChanges();
    await fixture.whenStable();

    expect(sessionTree.output().heightmapId).toBe('hm-002');
    expect(sessionTree.output().plan?.planId).toBe('plan-002');
  });

  // ── Bug regression: stale plan after profile change ──────────────────────

  it('changing profile invalidates the plan and triggers a fresh auto-compute', async () => {
    const fixture = TestBed.createComponent(WizardShellComponent);
    fixture.detectChanges();
    await fixture.whenStable();
    const cmp = fixture.componentInstance as unknown as {
      onFileSelected: (e: Event) => void;
      onHeightmapFileSelected: (e: Event) => void;
      renderPreview: () => void;
      onProfileSelected: (e: Event) => void;
    };
    cmp.onFileSelected({ target: { files: { item: () => new File(['x'], 'x.jpg') }, value: '' } } as unknown as Event);
    cmp.onHeightmapFileSelected({ target: { files: { item: () => new File(['x'], 'h.png') }, value: '' } } as unknown as Event);
    cmp.renderPreview();
    fixture.detectChanges();
    await fixture.whenStable();
    expect(sessionTree.output().plan?.planId).toBe('plan-001');

    apiMock.plan.mockReturnValue(of(PLAN_RESPONSE_V2));
    cmp.onProfileSelected({ target: { value: 'mopa_60w_steel' } } as unknown as Event);
    fixture.detectChanges();
    await fixture.whenStable();

    expect(sessionTree.pipeline().render.profileName).toBe('mopa_60w_steel');
    expect(sessionTree.output().plan?.planId).toBe('plan-002');
  });

  // ── Step 4: Exports ──────────────────────────────────────────────────────

  it('export.lbrn2 sends the current plan_id + heightmap_id', async () => {
    const fixture = TestBed.createComponent(WizardShellComponent);
    fixture.detectChanges();
    await fixture.whenStable();
    const cmp = fixture.componentInstance as unknown as {
      onFileSelected: (e: Event) => void;
      onHeightmapFileSelected: (e: Event) => void;
      renderPreview: () => void;
      exportLbrn2: () => void;
    };
    cmp.onFileSelected({ target: { files: { item: () => new File(['x'], 'x.jpg') }, value: '' } } as unknown as Event);
    cmp.onHeightmapFileSelected({ target: { files: { item: () => new File(['x'], 'h.png') }, value: '' } } as unknown as Event);
    cmp.renderPreview();
    fixture.detectChanges();
    await fixture.whenStable();
    cmp.exportLbrn2();
    expect(apiMock.exportLbrn2).toHaveBeenCalledWith(
      expect.objectContaining({ plan_id: 'plan-001', heightmap_id: 'hm-001' }),
    );
  });

  // ── Submit-the-bundle (replaces the dead Next button on step 5) ──────────

  it('submitBundle posts /export/bundle with all three formats when prereqs are met', async () => {
    const fixture = TestBed.createComponent(WizardShellComponent);
    fixture.detectChanges();
    await fixture.whenStable();
    const cmp = fixture.componentInstance as unknown as {
      onFileSelected: (e: Event) => void;
      onHeightmapFileSelected: (e: Event) => void;
      renderPreview: () => void;
      submitBundle: () => void;
      canSubmitBundle: () => boolean;
    };
    cmp.onFileSelected({ target: { files: { item: () => new File(['x'], 'x.jpg') }, value: '' } } as unknown as Event);
    cmp.onHeightmapFileSelected({ target: { files: { item: () => new File(['x'], 'h.png') }, value: '' } } as unknown as Event);
    cmp.renderPreview();
    fixture.detectChanges();
    await fixture.whenStable();
    expect(cmp.canSubmitBundle()).toBe(true);
    cmp.submitBundle();
    expect(apiMock.exportBundle).toHaveBeenCalledWith(
      expect.objectContaining({
        heightmap_id: 'hm-001',
        plan_id: 'plan-001',
        include_png: true,
        include_lbrn2: true,
        include_stl: true,
      }),
    );
  });

  it('submitBundle drops .lbrn2 from the request when the user unticks it', async () => {
    const fixture = TestBed.createComponent(WizardShellComponent);
    fixture.detectChanges();
    await fixture.whenStable();
    const cmp = fixture.componentInstance as unknown as {
      onFileSelected: (e: Event) => void;
      onHeightmapFileSelected: (e: Event) => void;
      renderPreview: () => void;
      toggleExport: (f: 'png' | 'lbrn2' | 'stl', e: Event) => void;
      submitBundle: () => void;
    };
    cmp.onFileSelected({ target: { files: { item: () => new File(['x'], 'x.jpg') }, value: '' } } as unknown as Event);
    cmp.onHeightmapFileSelected({ target: { files: { item: () => new File(['x'], 'h.png') }, value: '' } } as unknown as Event);
    cmp.renderPreview();
    fixture.detectChanges();
    await fixture.whenStable();
    cmp.toggleExport('lbrn2', { target: { checked: false } } as unknown as Event);
    cmp.submitBundle();
    expect(apiMock.exportBundle).toHaveBeenCalledWith(
      expect.objectContaining({ include_lbrn2: false }),
    );
    const arg = (apiMock.exportBundle as unknown as { mock: { calls: unknown[][] } }).mock.calls[0][0] as Record<string, unknown>;
    expect(arg['plan_id']).toBeUndefined();
  });

  it('canSubmitBundle is false when nothing has been rendered yet', () => {
    const fixture = TestBed.createComponent(WizardShellComponent);
    fixture.detectChanges();
    const cmp = fixture.componentInstance as unknown as { canSubmitBundle: () => boolean };
    expect(cmp.canSubmitBundle()).toBe(false);
  });
});
