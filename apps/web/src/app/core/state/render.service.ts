import { Injectable, inject } from '@angular/core';

import { ApiClientService } from '../api/api-client.service';
import { HeightmapSettings } from '../api/api-types';
import { BlobCacheService } from './blob-cache.service';
import { SessionTreeService } from './session-tree.service';

@Injectable({ providedIn: 'root' })
export class RenderService {
  private readonly apiClient = inject(ApiClientService);
  private readonly blobCache = inject(BlobCacheService);
  private readonly sessionTree = inject(SessionTreeService);

  render(): void {
    const state = this.sessionTree.state();
    if (!state.session.imageId) {
      return;
    }

    const { render } = state.pipeline;

    // Sculptok-only backend: depth comes from settings.external_heightmap_path
    // (set by upload / sculptok auto-pull). Render forwards only fields the
    // current backend accepts; everything else uses Pydantic defaults.
    this.apiClient
      .render({
        image_id: state.session.imageId,
        profile_name: render.profileName ?? undefined,
        settings: {
          input_clahe: true,
          input_clahe_clip: 3.0,
        } as Partial<HeightmapSettings> as HeightmapSettings,
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
              elapsedSeconds: response.elapsed_s,
            },
          }));
          this.sessionTree.pushHistory(
            'render:run',
            Math.round((response.elapsed_s ?? 0) * 1000),
          );
        },
        error: (err) => {
          const detail = err?.error?.detail ?? err?.message ?? 'Unknown error';
          this.sessionTree.addToast({ id: crypto.randomUUID(), severity: 'error', summary: 'Render failed', detail });
        },
      });
  }
}