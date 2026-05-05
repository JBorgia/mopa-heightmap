import { Injectable, inject } from '@angular/core';

import { ApiClientService } from '../api/api-client.service';
import { HeightmapSettings, InferenceConfig } from '../api/api-types';
import { BlobCacheService } from './blob-cache.service';
import { SessionTreeService } from './session-tree.service';

@Injectable({ providedIn: 'root' })
export class RenderService {
  private readonly apiClient = inject(ApiClientService);
  private readonly blobCache = inject(BlobCacheService);
  private readonly sessionTree = inject(SessionTreeService);

  setDetailBalance(detailBalance: number): void {
    this.sessionTree.patchState((current) => ({
      ...current,
      pipeline: {
        ...current.pipeline,
        render: {
          ...current.pipeline.render,
          detailBalance,
        },
      },
    }));
  }

  setMultires(multires: boolean): void {
    this.sessionTree.patchState((current) => ({
      ...current,
      pipeline: {
        ...current.pipeline,
        render: {
          ...current.pipeline.render,
          multires,
        },
      },
    }));
  }

  setRelief(relief: number): void {
    this.sessionTree.patchState((current) => ({
      ...current,
      pipeline: {
        ...current.pipeline,
        render: {
          ...current.pipeline.render,
          relief,
        },
      },
    }));
  }

  render(): void {
    const state = this.sessionTree.state();
    if (!state.session.imageId) {
      return;
    }

    const { render, advanced } = state.pipeline;

    this.apiClient
      .render({
        image_id: state.session.imageId,
        profile_name: render.profileName ?? undefined,
        // Pydantic supplies defaults for every omitted field on the server.
        settings: {
          detail_mode: render.detailBalance > 0 ? 'luminance' : 'off',
          detail_strength: render.detailBalance,
          edge_refine: true,
          edge_refine_diameter: 9,
          edge_refine_sigma_color: 0.1,
          edge_refine_sigma_space: 8.0,
          sharpen: advanced.sharpen,
          smooth_strength: advanced.bilateralStrength,
          contrast: 0.5 + render.relief,
          input_clahe: true,
          input_clahe_clip: 3.0,
        } as unknown as HeightmapSettings,
        inference: {
          with_flip_aug: render.multires,
        } as unknown as InferenceConfig,
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