import { Injectable, computed, inject, signal } from '@angular/core';

import { ApiClientService } from '../api/api-client.service';
import type { HeightmapSettings, TargetPresetSummary } from '../api/api-types';
import { RenderService } from './render.service';
import { SessionTreeService } from './session-tree.service';

/**
 * Loads target-object presets (coin / signet_ring / pendant / plaque /
 * portrait) and applies one to the live HeightmapSettings.
 *
 * The presets themselves live in ``profiles/targets/*.yaml`` server-side.
 * Applying a preset only sets the fields the preset defines, leaving
 * everything else at its current value.
 */
@Injectable({ providedIn: 'root' })
export class TargetService {
  private readonly apiClient = inject(ApiClientService);
  private readonly renderService = inject(RenderService);
  private readonly sessionTree = inject(SessionTreeService);

  readonly presets = signal<TargetPresetSummary[]>([]);
  readonly active = signal<string | null>(null);
  readonly activeShape = signal<string | null>(null);

  /** Physical dimensions of the engraving area in mm. Set by preset or manual input. */
  readonly printWidthMm = signal<number>(0);
  readonly printHeightMm = signal<number>(0);

  readonly activePreset = computed(() => {
    const name = this.active();
    return name ? (this.presets().find(p => p.name === name) ?? null) : null;
  });

  loadPresets(): void {
    this.apiClient.listTargets().subscribe({
      next: (rows) => this.presets.set(rows),
      error: () => this.presets.set([]),
    });
  }

  apply(name: string): void {
    const preset = this.presets().find((p) => p.name === name);
    if (!preset) return;
    // Always-applied fields from the preset summary.
    this.renderService.patchSettings('polarity_invert', preset.polarity_invert);
    if (preset.name === 'plaque') {
      this.renderService.patchSettings('subject_mask_enabled', false);
      this.renderService.patchSettings('input_auto_crop', false);
    } else {
      this.renderService.patchSettings('subject_mask_enabled', true);
      this.renderService.patchSettings('input_auto_crop', true);
    }
    if (preset.print_height_mm > 0) {
      this.renderService.patchSettings(
        'input_auto_crop_aspect',
        preset.print_width_mm / preset.print_height_mm,
      );
    }

    // Metal blanks (coin, signet ring, plaque) almost always have specular
    // highlights that Sculptok misreads as raised depth — enable removal
    // proactively so the default path produces a clean heightmap.
    const metalTargets = new Set(['coin', 'signet_ring', 'plaque']);
    if (metalTargets.has(preset.name)) {
      this.renderService.patchSettings('input_remove_specular', true);
    }

    this.active.set(name);
    this.activeShape.set(preset.default_shape ?? 'rectangle');
    this.printWidthMm.set(preset.print_width_mm);
    this.printHeightMm.set(preset.print_height_mm);
    this.syncZoneGeometry();
    this.sessionTree.pushHistory(`target:apply:${name}`);

    const metalNote = metalTargets.has(preset.name)
      ? ' · Specular removal enabled (metal surface detected)'
      : '';
    this.sessionTree.addToast({
      id: crypto.randomUUID(),
      severity: 'info',
      summary: 'Target preset applied',
      detail: `${preset.display_name} — ${preset.print_width_mm}×${preset.print_height_mm} mm${preset.polarity_invert ? ', polarity inverted' : ''}${metalNote}`,
    });
  }

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
