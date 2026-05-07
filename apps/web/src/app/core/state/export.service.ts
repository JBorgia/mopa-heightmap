import { Injectable, inject, signal } from '@angular/core';

import { ApiClientService } from '../api/api-client.service';
import { SessionTreeService } from './session-tree.service';

export const EXPORT_PNG_FILENAME = 'heightmap.png';
// Server emits a zip bundle (`.lbrn2 project + per-pass PNGs`). User must
// unzip before LightBurn opens it — naming the download `.lbrn2` was a
// silent corruption bug.
export const EXPORT_LBRN2_FILENAME = 'project.lbrn2.zip';
export const EXPORT_STL_FILENAME = 'heightmap.stl';
export const EXPORT_BUNDLE_FILENAME = 'mopa_export.zip';
export const EXPORT_STL_DEFAULT_Z_SCALE_MM = 5.0;
export const EXPORT_STL_DEFAULT_BASE_THICKNESS_MM = 2.0;

@Injectable({ providedIn: 'root' })
export class ExportService {
  private readonly apiClient = inject(ApiClientService);
  private readonly sessionTree = inject(SessionTreeService);

  /** True while a /export/bundle request is outstanding — drives the wizard
   * Submit button's "Bundling…" label and prevents duplicate downloads. */
  readonly bundleInFlight = signal(false);

  exportPng(): void {
    const state = this.sessionTree.state();
    if (!state.output.heightmapId) {
      return;
    }

    this.apiClient.exportPng({ heightmap_id: state.output.heightmapId, bit_depth: 16 }).subscribe({
      next: (blob) => {
        this._triggerDownload(blob, EXPORT_PNG_FILENAME);
        this.sessionTree.pushHistory('export:png');
      },
      error: (err) => {
        const detail = err?.error?.detail ?? err?.message ?? 'Unknown error';
        this.sessionTree.addToast({ id: crypto.randomUUID(), severity: 'error', summary: 'PNG export failed', detail });
      },
    });
  }

  exportLbrn2(): void {
    const state = this.sessionTree.state();
    if (!state.output.plan || !state.output.heightmapId) {
      return;
    }

    this.apiClient
      .exportLbrn2({
        plan_id: state.output.plan.planId,
        heightmap_id: state.output.heightmapId,
        profile_name: state.pipeline.render.profileName ?? undefined,
      })
      .subscribe({
        next: (blob) => {
          this._triggerDownload(blob, EXPORT_LBRN2_FILENAME);
          this.sessionTree.pushHistory('export:lbrn2');
        },
        error: (err) => {
          const detail = err?.error?.detail ?? err?.message ?? 'Unknown error';
          this.sessionTree.addToast({ id: crypto.randomUUID(), severity: 'error', summary: 'LightBurn export failed', detail });
        },
      });
  }

  exportStl(): void {
    const state = this.sessionTree.state();
    if (!state.output.heightmapId) {
      return;
    }

    this.apiClient
      .exportStl({
        heightmap_id: state.output.heightmapId,
        z_scale_mm: EXPORT_STL_DEFAULT_Z_SCALE_MM,
        base_thickness_mm: EXPORT_STL_DEFAULT_BASE_THICKNESS_MM,
      })
      .subscribe({
        next: (blob) => {
          this._triggerDownload(blob, EXPORT_STL_FILENAME);
          this.sessionTree.pushHistory('export:stl');
        },
        error: (err) => {
          const detail = err?.error?.detail ?? err?.message ?? 'Unknown error';
          this.sessionTree.addToast({ id: crypto.randomUUID(), severity: 'error', summary: 'STL export failed', detail });
        },
      });
  }

  /** Drives the wizard's Submit action — bundles the selected formats into
   * one zip and triggers a single download. */
  exportBundle(opts: { png: boolean; lbrn2: boolean; stl: boolean }): void {
    const state = this.sessionTree.state();
    if (!state.output.heightmapId) {
      return;
    }
    if (!opts.png && !opts.lbrn2 && !opts.stl) {
      return;
    }
    if (this.bundleInFlight()) {
      return;
    }

    this.bundleInFlight.set(true);
    // Forward every reference artifact we know about — the server bundles
    // them unconditionally when present (mask, source photo, sculptok
    // input, profile YAML). The user shouldn't have to re-run the wizard
    // because they didn't pre-tick a "include mask" box that wasn't
    // surfaced in the UI.
    const userMaskId = state.pipeline.mask.maskId ?? undefined;
    const renderMaskId = state.output.renderMaskId ?? undefined;
    this.apiClient
      .exportBundle({
        heightmap_id: state.output.heightmapId,
        plan_id: opts.lbrn2 ? state.output.plan?.planId ?? undefined : undefined,
        profile_name: state.pipeline.render.profileName ?? undefined,
        include_png: opts.png,
        include_lbrn2: opts.lbrn2,
        include_stl: opts.stl,
        z_scale_mm: EXPORT_STL_DEFAULT_Z_SCALE_MM,
        base_thickness_mm: EXPORT_STL_DEFAULT_BASE_THICKNESS_MM,
        image_id: state.session.imageId ?? undefined,
        sculptok_input_id: state.output.sculptokInputId ?? undefined,
        // Prefer the user-driven mask (page 1) over the render-time one;
        // they both cover the same shape but the page-1 mask is what the
        // user explicitly intended to ship as the deliverable.
        subject_mask_id: userMaskId ?? renderMaskId,
      })
      .subscribe({
        next: (blob) => {
          this._triggerDownload(blob, EXPORT_BUNDLE_FILENAME);
          const formats = [opts.png && 'PNG', opts.lbrn2 && '.lbrn2', opts.stl && '.stl'].filter(Boolean).join(' + ');
          this.sessionTree.pushHistory(`export:bundle:${formats}`);
          this.sessionTree.addToast({
            id: crypto.randomUUID(),
            severity: 'success',
            summary: 'Bundle ready',
            detail: `${formats} downloaded as ${EXPORT_BUNDLE_FILENAME}`,
          });
          this.bundleInFlight.set(false);
        },
        error: (err) => {
          const detail = err?.error?.detail ?? err?.message ?? 'Unknown error';
          this.sessionTree.addToast({ id: crypto.randomUUID(), severity: 'error', summary: 'Bundle export failed', detail });
          this.bundleInFlight.set(false);
        },
      });
  }

  private _triggerDownload(blob: Blob, filename: string): void {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }
}