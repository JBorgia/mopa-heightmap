import { Injectable, inject, signal } from '@angular/core';

import type { HeightmapSettings } from '../api/api-types';
import { RenderService } from './render.service';

/**
 * Holds the physical blank geometry (shape + print size in mm) and mirrors
 * it into HeightmapSettings so zone overlays and the crop overlay stay in
 * sync with what the user is actually engraving on.
 */
@Injectable({ providedIn: 'root' })
export class TargetService {
  private readonly renderService = inject(RenderService);

  readonly activeShape = signal<string | null>(null);

  /** Physical dimensions of the engraving area in mm. Set by manual input
   * or restored from a saved preset. */
  readonly printWidthMm = signal<number>(0);
  readonly printHeightMm = signal<number>(0);

  setShape(shape: string): void {
    this.activeShape.set(shape);
    this.syncZoneGeometry();
  }

  setDimensions(widthMm: number, heightMm: number): void {
    this.printWidthMm.set(widthMm);
    this.printHeightMm.set(heightMm);
    this.syncZoneGeometry();
  }

  /**
   * Mirror the blank's physical geometry into HeightmapSettings so zone
   * overlays (field / rim / border) know where to draw. Without this the
   * user would have to enter the blank size twice and the overlays would
   * silently no-op at zone_width_mm = 0.
   */
  private syncZoneGeometry(): void {
    const w = this.printWidthMm();
    const h = this.printHeightMm();
    if (w > 0 && h > 0) {
      this.renderService.patchSettings('zone_width_mm', w);
      this.renderService.patchSettings('zone_height_mm', h);
    }
    const shape = this.activeShape();
    if (shape) {
      this.renderService.patchSettings('zone_shape', shape as HeightmapSettings['zone_shape']);
    }
  }
}
