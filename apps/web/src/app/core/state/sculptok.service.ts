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
    this.apiClient
      .sculptokGenerate({
        image_id: imageId,
        style: opts.style ?? 'pro',
        version: opts.version ?? '1.5',
        draw_hd: opts.draw_hd ?? '2k',
      })
      .subscribe({
        next: (resp) => {
          // Server-side path → settings.external_heightmap_path. The
          // /render endpoint reads this to load the heightmap.
          this.renderService.patchSettings('external_heightmap_path', resp.heightmap_path);
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
