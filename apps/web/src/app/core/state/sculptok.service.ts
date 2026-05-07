import { Injectable, inject, signal } from '@angular/core';

import { ApiClientService } from '../api/api-client.service';
import type { SculptokCreditsResponse } from '../api/api-types';
import { RenderService } from './render.service';
import { SessionTreeService } from './session-tree.service';

/**
 * Drives the sculptok auto-pull flow:
 *
 *   * ``loadCredits()`` — refresh the credit balance signal so the UI can
 *     show "configured / N credits remaining" or a clear "not configured"
 *     state.
 *   * ``generate()`` — call POST /sculptok/generate against the current
 *     uploaded image, point ``settings.external_heightmap_path`` at the
 *     resulting server-side path, and refresh the credit balance.
 */
@Injectable({ providedIn: 'root' })
export class SculptokService {
  private readonly apiClient = inject(ApiClientService);
  private readonly renderService = inject(RenderService);
  private readonly sessionTree = inject(SessionTreeService);

  readonly credits = signal<SculptokCreditsResponse | null>(null);
  readonly inFlight = signal<boolean>(false);

  loadCredits(): void {
    this.apiClient.sculptokCredits().subscribe({
      next: (resp) => this.credits.set(resp),
      error: () => this.credits.set({ configured: false, balance: null, cost_pro_2k: 15, cost_pro_4k: 30, cost_normal: 10 }),
    });
  }

  /** Upload a hand-supplied heightmap PNG; sets external_heightmap_path. */
  uploadHeightmap(file: File): void {
    this.inFlight.set(true);
    this.apiClient.uploadHeightmap(file).subscribe({
      next: (resp) => {
        this.renderService.patchSettings('external_heightmap_path', resp.heightmap_path);
        // The server detects Sculptok side-by-side composites (depth map +
        // render preview joined horizontally) and crops to the depth-map
        // half. Surface a warn toast — the user shipped one PNG and got
        // a half-as-wide PNG back; they should know.
        if (resp.auto_cropped) {
          this.sessionTree.addToast({
            id: crypto.randomUUID(),
            severity: 'warn',
            summary: 'Side-by-side composite detected',
            detail:
              `Cropped to the depth-map half (${resp.width}×${resp.height} px). ` +
              `If the result looks wrong, re-upload the depth-map-only export ` +
              `from Sculptok rather than the comparison preview.`,
          });
        } else {
          this.sessionTree.addToast({
            id: crypto.randomUUID(),
            severity: 'success',
            summary: 'Heightmap uploaded',
            detail: `${resp.width}×${resp.height} px`,
          });
        }
        this.sessionTree.pushHistory('heightmap:upload');
        this.inFlight.set(false);
      },
      error: (err) => {
        const detail = err?.error?.detail ?? err?.message ?? 'Unknown error';
        this.sessionTree.addToast({
          id: crypto.randomUUID(),
          severity: 'error',
          summary: 'Heightmap upload failed',
          detail,
        });
        this.inFlight.set(false);
      },
    });
  }

  generate(opts: {
    style?: 'normal' | 'portrait' | 'sketch' | 'pro';
    version?: '1.0' | '1.5';
    draw_hd?: '2k' | '4k';
  } = {}): void {
    const state = this.sessionTree.state();
    const imageId = state.session.imageId;
    if (!imageId) {
      this.sessionTree.addToast({
        id: crypto.randomUUID(),
        severity: 'warn',
        summary: 'Sculptok',
        detail: 'Upload an image first.',
      });
      return;
    }

    this.inFlight.set(true);
    // Forward the current heightmap settings so the server can apply
    // pre-sculptok prep (CLAHE / denoise / specular / auto-orient / auto-
    // crop / bg-replace) BEFORE uploading. Without this the prep toggles
    // would be cosmetic — sculptok would still see the raw photo.
    this.apiClient
      .sculptokGenerate({
        image_id: imageId,
        style: opts.style ?? 'pro',
        version: opts.version ?? '1.5',
        draw_hd: opts.draw_hd ?? '2k',
        settings: state.pipeline.settings,
      })
      .subscribe({
        next: (resp) => {
          this.renderService.patchSettings('external_heightmap_path', resp.heightmap_path);
          // Surface the prepped photo + (optional) subject mask so the
          // wizard's preview pane shows what sculptok actually saw.
          this.sessionTree.patchState((current) => ({
            ...current,
            output: {
              ...current.output,
              sculptokInputId: resp.sculptok_input_id ?? null,
              renderMaskId: resp.subject_mask_id ?? current.output.renderMaskId,
            },
          }));
          this.credits.set({
            configured: true,
            balance: resp.credits_remaining,
            cost_pro_2k: 15,
            cost_pro_4k: 30,
            cost_normal: 10,
          });
          this.sessionTree.addToast({
            id: crypto.randomUUID(),
            severity: 'success',
            summary: 'Sculptok generated',
            detail: `Used ${resp.credits_used} credits — ${resp.credits_remaining} remaining`,
          });
          this.sessionTree.pushHistory('sculptok:generate');
          this.inFlight.set(false);
        },
        error: (err) => {
          const detail = err?.error?.detail ?? err?.message ?? 'Unknown error';
          this.sessionTree.addToast({
            id: crypto.randomUUID(),
            severity: 'error',
            summary: 'Sculptok generate failed',
            detail,
          });
          this.inFlight.set(false);
        },
      });
  }
}
