import { Injectable, inject } from '@angular/core';

import { ClickMaskRequest } from '../api/api-types';
import { ApiClientService } from '../api/api-client.service';
import { SessionTreeService } from './session-tree.service';
import { ClickerKey, MaskBackend } from './studio-state';

@Injectable({ providedIn: 'root' })
export class MaskService {
  private readonly apiClient = inject(ApiClientService);
  private readonly sessionTree = inject(SessionTreeService);

  setBackend(backend: MaskBackend): void {
    this.sessionTree.patchState((current) => ({
      ...current,
      pipeline: {
        ...current.pipeline,
        mask: {
          ...current.pipeline.mask,
          backend,
        },
      },
    }));
  }

  setClickerKey(clickerKey: ClickerKey): void {
    this.sessionTree.patchState((current) => ({
      ...current,
      pipeline: {
        ...current.pipeline,
        mask: {
          ...current.pipeline.mask,
          clickerKey,
        },
      },
    }));
  }

  setEdgeSoftness(edgeSoftness: number): void {
    this.sessionTree.patchState((current) => ({
      ...current,
      pipeline: {
        ...current.pipeline,
        mask: {
          ...current.pipeline.mask,
          edgeSoftness,
        },
      },
    }));
  }

  createMask(): void {
    const state = this.sessionTree.state();
    if (!state.session.imageId) {
      return;
    }

    this.apiClient
      .createMask({
        image_id: state.session.imageId,
        backend: state.pipeline.mask.backend,
        edge_softness: state.pipeline.mask.edgeSoftness,
      })
      .subscribe({
        next: (response) => {
          this.sessionTree.patchState((current) => ({
            ...current,
            pipeline: {
              ...current.pipeline,
              mask: {
                ...current.pipeline.mask,
                maskId: response.mask_id,
                coveragePct: response.coverage_pct,
              },
            },
          }));
          this.sessionTree.pushHistory('mask:create');
        },
        error: (err) => {
          const detail = err?.error?.detail ?? err?.message ?? 'Unknown error';
          this.sessionTree.addToast({ id: crypto.randomUUID(), severity: 'error', summary: 'Mask failed', detail });
        },
      });
  }

  /**
   * BUG-2 fix: click-refine uses `clicker_key` (ClickerKey), never `backend` (MaskBackend).
   * The clicker registry and the mask backend registry are separate; sharing the control
   * would cause KeyError when a non-clicker backend (birefnet, rembg) is selected.
   */
  clickRefine(x: number, y: number, label: 'positive' | 'negative' = 'positive'): void {
    const state = this.sessionTree.state();
    if (!state.session.imageId) {
      return;
    }

    const request: ClickMaskRequest = {
      image_id: state.session.imageId,
      mask_id: state.pipeline.mask.maskId ?? undefined,
      x,
      y,
      label,
      clicker_key: state.pipeline.mask.clickerKey,
      tolerance: 0.08,
      max_fraction: 0.6,
    };

    this.apiClient.clickMask(request).subscribe({
      next: (response) => {
        this.sessionTree.patchState((current) => ({
          ...current,
          pipeline: {
            ...current.pipeline,
            mask: {
              ...current.pipeline.mask,
              maskId: response.mask_id,
              coveragePct: response.coverage_pct,
            },
          },
        }));
        this.sessionTree.pushHistory(`mask:click-refine:${label}`);
      },
      error: (err) => {
        const detail = err?.error?.detail ?? err?.message ?? 'Unknown error';
        this.sessionTree.addToast({ id: crypto.randomUUID(), severity: 'error', summary: 'Refine failed', detail });
      },
    });
  }
}