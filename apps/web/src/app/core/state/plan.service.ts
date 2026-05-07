import { Injectable, inject, signal } from '@angular/core';

import { ApiClientService } from '../api/api-client.service';
import { SessionTreeService } from './session-tree.service';

@Injectable({ providedIn: 'root' })
export class PlanService {
  private readonly apiClient = inject(ApiClientService);
  private readonly sessionTree = inject(SessionTreeService);

  readonly inFlight = signal(false);

  computePlan(): void {
    const output = this.sessionTree.output();
    const session = this.sessionTree.session();
    const pipeline = this.sessionTree.pipeline();

    if (!output.heightmapId || !session.imageId) {
      return;
    }
    if (this.inFlight()) {
      return;
    }

    this.inFlight.set(true);
    this.apiClient
      .plan({
        image_id: session.imageId,
        heightmap_id: output.heightmapId,
        profile_name: pipeline.render.profileName ?? undefined,
      })
      .subscribe({
        next: (response) => {
          this.sessionTree.patchState((current) => ({
            ...current,
            output: {
              ...current.output,
              plan: {
                planId: response.plan_id,
                passes: response.passes.map((p) => ({
                  passNumber: p.pass_number,
                  label: p.label,
                  depthUm: p.depth_um,
                  colorHex: p.color_hex,
                })),
              },
            },
          }));
          this.sessionTree.pushHistory(`plan:compute:${response.passes.length} passes`);
          this.inFlight.set(false);
        },
        error: (err) => {
          const detail = err?.error?.detail ?? err?.message ?? 'Unknown error';
          this.sessionTree.addToast({ id: crypto.randomUUID(), severity: 'error', summary: 'Pass plan failed', detail });
          this.inFlight.set(false);
        },
      });
  }
}
