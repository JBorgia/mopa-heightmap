import { Injectable, inject, signal } from '@angular/core';

import { ApiClientService } from '../api/api-client.service';
import { HeightmapSettings } from '../api/api-types';
import { BlobCacheService } from './blob-cache.service';
import { SessionTreeService } from './session-tree.service';

@Injectable({ providedIn: 'root' })
export class RenderService {
  private readonly apiClient = inject(ApiClientService);
  private readonly blobCache = inject(BlobCacheService);
  private readonly sessionTree = inject(SessionTreeService);

  /** True while a /render request is outstanding — drives button feedback. */
  readonly inFlight = signal(false);

  /**
   * Generic settings patch — set a single ``HeightmapSettings`` field on
   * the live state. UI controls bind to this so we don't proliferate
   * one-shot setters per knob.
   */
  patchSettings<K extends keyof HeightmapSettings>(
    key: K,
    value: HeightmapSettings[K],
  ): void {
    this.sessionTree.patchState((current) => ({
      ...current,
      pipeline: {
        ...current.pipeline,
        settings: {
          ...current.pipeline.settings,
          [key]: value,
        },
      },
    }));
  }

  render(): void {
    const state = this.sessionTree.state();
    if (!state.session.imageId) {
      return;
    }
    if (this.inFlight()) {
      return; // ignore double-clicks while a request is outstanding
    }

    const { render, settings } = state.pipeline;
    this.inFlight.set(true);

    // Sculptok-only backend: depth comes from settings.external_heightmap_path
    // (set by upload / sculptok auto-pull). The state's `settings` object
    // mirrors HeightmapSettings 1:1, so we forward it verbatim.
    this.apiClient
      .render({
        image_id: state.session.imageId,
        profile_name: render.profileName ?? undefined,
        settings,
      })
      .subscribe({
        next: (response) => {
          this.blobCache.get(response.image_hash);
          this.sessionTree.patchState((current) => ({
            ...current,
            session: {
              ...current.session,
              imageHash: response.image_hash,
            },
            output: {
              ...current.output,
              heightmapId: response.heightmap_id,
              previewId: response.preview_id,
              conditionedId: response.conditioned_id ?? null,
              renderMaskId: response.render_mask_id ?? null,
              // Any prior plan was computed for the previous heightmap.
              // Clearing it forces the wizard's auto-compute to recompute.
              plan: null,
              elapsedSeconds: response.elapsed_s,
            },
          }));
          this.sessionTree.pushHistory(
            'render:run',
            Math.round((response.elapsed_s ?? 0) * 1000),
          );
          this.inFlight.set(false);
        },
        error: (err) => {
          const detail = err?.error?.detail ?? err?.message ?? 'Unknown error';
          this.sessionTree.addToast({ id: crypto.randomUUID(), severity: 'error', summary: 'Render failed', detail });
          this.inFlight.set(false);
        },
      });
  }
}
