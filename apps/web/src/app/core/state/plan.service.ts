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
    // When the blank geometry is configured, forward its shape so zone
    // masks are cut for the actual blank instead of the profile default.
    const shapeOverride =
      (pipeline.settings.zone_width_mm ?? 0) > 0 && (pipeline.settings.zone_height_mm ?? 0) > 0
        ? pipeline.settings.zone_shape
        : undefined;
    this.apiClient
      .plan({
        image_id: session.imageId,
        heightmap_id: output.heightmapId,
        profile_name: pipeline.render.profileName ?? undefined,
        settings: pipeline.settings,
        shape_override: shapeOverride,
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
                estimatedRuntimeS: response.estimated_runtime_s ?? 0,
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
