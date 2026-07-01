import { Injectable, inject, signal } from '@angular/core';

import { ClickMaskRequest } from '../api/api-types';
import { ApiClientService } from '../api/api-client.service';
import { SessionTreeService } from './session-tree.service';
import { ClickerKey, MaskBackend } from './studio-state';

@Injectable({ providedIn: 'root' })
export class MaskService {
  private readonly apiClient = inject(ApiClientService);
  private readonly sessionTree = inject(SessionTreeService);

  /** True while any mask API call (create or click-refine) is in flight. */
  readonly inFlight = signal(false);

  /**
   * The input key (imageId|backend|edgeSoftness) used when the current maskId
   * was produced. Compared against the live key to decide if the mask is still
   * fresh. Null when no mask has been created this session.
   */
  readonly lastMaskedKey = signal<string | null>(null);

  private maskKey(): string {
    const { session, pipeline } = this.sessionTree.state();
    return `${session.imageId}|${pipeline.mask.backend}|${pipeline.mask.edgeSoftness}`;
  }

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
    if (!state.session.imageId) return;
    if (this.inFlight()) return;

    const key = this.maskKey();
    this.inFlight.set(true);

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
          this.lastMaskedKey.set(key);
          this.sessionTree.pushHistory('mask:create');
          this.inFlight.set(false);
        },
        error: (err) => {
          const detail = err?.error?.detail ?? err?.message ?? 'Unknown error';
          this.sessionTree.addToast({ id: crypto.randomUUID(), severity: 'error', summary: 'Mask failed', detail });
          this.inFlight.set(false);
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
    if (!state.session.imageId) return;
    if (this.inFlight()) return;

    this.inFlight.set(true);

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
        this.inFlight.set(false);
      },
      error: (err) => {
        const detail = err?.error?.detail ?? err?.message ?? 'Unknown error';
        this.sessionTree.addToast({ id: crypto.randomUUID(), severity: 'error', summary: 'Refine failed', detail });
        this.inFlight.set(false);
      },
    });
  }
}