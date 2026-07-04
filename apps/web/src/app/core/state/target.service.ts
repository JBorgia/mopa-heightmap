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
    this.active.set(name);
    this.activeShape.set(preset.default_shape ?? 'rectangle');
    this.sessionTree.pushHistory(`target:apply:${name}`);
    this.sessionTree.addToast({
      id: crypto.randomUUID(),
      severity: 'info',
      summary: 'Target preset applied',
      detail: `${preset.display_name} — ${preset.print_width_mm}×${preset.print_height_mm} mm${preset.polarity_invert ? ', polarity inverted' : ''}`,
    });
  }

  setShape(shape: string): void {
    this.activeShape.set(shape);
  }

  /** Helper used by the UI to read a single field from the active preset. */
  activeWidthMm(): number | null {
    const name = this.active();
    if (!name) return null;
    const preset = this.presets().find((p) => p.name === name);
    return preset?.print_width_mm ?? null;
  }

  activeHeightMm(): number | null {
    const name = this.active();
    if (!name) return null;
    const preset = this.presets().find((p) => p.name === name);
    return preset?.print_height_mm ?? null;
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  private _unused(_settings: HeightmapSettings): void {}
}
