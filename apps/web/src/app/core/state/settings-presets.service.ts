import { Injectable, inject, signal } from '@angular/core';

import { HeightmapSettings } from '../api/api-types';
import { SessionTreeService } from './session-tree.service';
import { TargetService } from './target.service';

/** localStorage key for the user's saved settings presets. */
export const USER_PRESETS_STORAGE_KEY = 'mopa-heightmap.user-presets';
/** Cap so a runaway save loop can't bloat localStorage. */
export const USER_PRESETS_MAX = 24;

export interface UserPreset {
  name: string;
  savedAt: string; // ISO date
  /** Blank shape at save time (targetService.activeShape). */
  shape: string | null;
  /** Snapshot of HeightmapSettings minus session-volatile fields. */
  settings: Partial<HeightmapSettings>;
}

/**
 * User-defined settings presets, persisted to localStorage.
 *
 * Replaces the old server-defined "Quick presets" (coin / signet / plaque
 * chips): instead of opinionated built-ins, the user saves named snapshots
 * of their own dialled-in settings and re-applies them per job. The *live*
 * settings already persist across sessions via the studio-state
 * localStorage serializer — presets are for switching between jobs.
 */
@Injectable({ providedIn: 'root' })
export class SettingsPresetsService {
  private readonly sessionTree = inject(SessionTreeService);
  private readonly targetService = inject(TargetService);

  readonly presets = signal<UserPreset[]>(this._load());

  /** Save the current settings + blank shape under ``name`` (overwrites
   * an existing preset with the same name). */
  saveCurrent(name: string): boolean {
    const trimmed = name.trim();
    if (!trimmed) {
      return false;
    }
    // external_heightmap_path points at a temp file from THIS session —
    // restoring it later would silently render a stale (or missing) depth
    // map, so it never belongs in a preset.
    const { external_heightmap_path: _omit, ...settings } =
      this.sessionTree.pipeline().settings;
    const preset: UserPreset = {
      name: trimmed,
      savedAt: new Date().toISOString(),
      shape: this.targetService.activeShape(),
      settings,
    };
    const rest = this.presets().filter((p) => p.name !== trimmed);
    this.presets.set([preset, ...rest].slice(0, USER_PRESETS_MAX));
    this._persist();
    return true;
  }

  /** Merge the named preset into the live settings and restore its blank
   * shape. Returns false when the preset doesn't exist. */
  apply(name: string): boolean {
    const preset = this.presets().find((p) => p.name === name);
    if (!preset) {
      return false;
    }
    this.sessionTree.patchState((current) => ({
      ...current,
      pipeline: {
        ...current.pipeline,
        settings: {
          ...current.pipeline.settings,
          ...preset.settings,
        },
      },
    }));
    // Restore the blank geometry signals so the step-1 inputs reflect the
    // preset (setShape/setDimensions also re-sync zone geometry).
    if (preset.shape) {
      this.targetService.setShape(preset.shape);
    }
    const w = preset.settings.zone_width_mm ?? 0;
    const h = preset.settings.zone_height_mm ?? 0;
    if (w > 0 && h > 0) {
      this.targetService.setDimensions(w, h);
    }
    return true;
  }

  remove(name: string): void {
    this.presets.set(this.presets().filter((p) => p.name !== name));
    this._persist();
  }

  private _load(): UserPreset[] {
    try {
      const raw = globalThis.localStorage?.getItem(USER_PRESETS_STORAGE_KEY);
      if (!raw) {
        return [];
      }
      const parsed = JSON.parse(raw) as unknown;
      if (!Array.isArray(parsed)) {
        return [];
      }
      return parsed.filter(
        (p): p is UserPreset =>
          !!p && typeof p === 'object' &&
          typeof (p as UserPreset).name === 'string' &&
          typeof (p as UserPreset).settings === 'object',
      );
    } catch {
      return [];
    }
  }

  private _persist(): void {
    try {
      globalThis.localStorage?.setItem(
        USER_PRESETS_STORAGE_KEY,
        JSON.stringify(this.presets()),
      );
    } catch {
      // Quota / privacy mode — presets stay in memory for the session.
    }
  }
}
