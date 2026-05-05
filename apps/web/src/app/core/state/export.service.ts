import { Injectable, inject } from '@angular/core';

import { ApiClientService } from '../api/api-client.service';
import { SessionTreeService } from './session-tree.service';

export const EXPORT_PNG_FILENAME = 'heightmap.png';
export const EXPORT_LBRN2_FILENAME = 'project.lbrn2';
export const EXPORT_STL_FILENAME = 'heightmap.stl';
export const EXPORT_STL_DEFAULT_Z_SCALE_MM = 5.0;
export const EXPORT_STL_DEFAULT_BASE_THICKNESS_MM = 2.0;

@Injectable({ providedIn: 'root' })
export class ExportService {
  private readonly apiClient = inject(ApiClientService);
  private readonly sessionTree = inject(SessionTreeService);

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

  private _triggerDownload(blob: Blob, filename: string): void {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }
}