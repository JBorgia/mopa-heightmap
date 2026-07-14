/**
 * Unit tests for the user-defined settings presets (localStorage-backed).
 */
import { TestBed } from '@angular/core/testing';
import { describe, it, expect, beforeEach } from 'vitest';

import { ApiClientService } from '../api/api-client.service';
import { SessionTreeService } from './session-tree.service';
import {
  SettingsPresetsService,
  USER_PRESETS_STORAGE_KEY,
  USER_PRESETS_MAX,
} from './settings-presets.service';
import { TargetService } from './target.service';

describe('SettingsPresetsService', () => {
  let service: SettingsPresetsService;
  let sessionTree: SessionTreeService;
  let targetService: TargetService;

  beforeEach(() => {
    localStorage.removeItem(USER_PRESETS_STORAGE_KEY);
    TestBed.configureTestingModule({
      providers: [{ provide: ApiClientService, useValue: {} }],
    });
    sessionTree = TestBed.inject(SessionTreeService);
    sessionTree.reset();
    targetService = TestBed.inject(TargetService);
    service = TestBed.inject(SettingsPresetsService);
  });

  it('storage key and cap are stable', () => {
    expect(USER_PRESETS_STORAGE_KEY).toBe('mopa-heightmap.user-presets');
    expect(USER_PRESETS_MAX).toBe(24);
  });

  it('saveCurrent snapshots settings + shape and persists to localStorage', () => {
    targetService.setShape('hexagon');
    targetService.setDimensions(35, 35);
    sessionTree.patchState((c) => ({
      ...c,
      pipeline: {
        ...c.pipeline,
        settings: {
          ...c.pipeline.settings,
          rim_pattern: 'denticled' as const,
          border_pattern: 'laurel' as const,
        },
      },
    }));

    expect(service.saveCurrent('  My coin  ')).toBe(true);
    const stored = JSON.parse(localStorage.getItem(USER_PRESETS_STORAGE_KEY)!);
    expect(stored).toHaveLength(1);
    expect(stored[0].name).toBe('My coin');
    expect(stored[0].shape).toBe('hexagon');
    expect(stored[0].settings.rim_pattern).toBe('denticled');
    expect(stored[0].settings.border_pattern).toBe('laurel');
  });

  it('saveCurrent excludes the session-volatile heightmap path', () => {
    sessionTree.patchState((c) => ({
      ...c,
      pipeline: {
        ...c.pipeline,
        settings: {
          ...c.pipeline.settings,
          external_heightmap_path: '/tmp/session-only.png',
        },
      },
    }));
    service.saveCurrent('no-path');
    const stored = JSON.parse(localStorage.getItem(USER_PRESETS_STORAGE_KEY)!);
    expect(stored[0].settings.external_heightmap_path).toBeUndefined();
  });

  it('saveCurrent rejects blank names and overwrites same-name presets', () => {
    expect(service.saveCurrent('   ')).toBe(false);
    service.saveCurrent('dup');
    service.saveCurrent('dup');
    expect(service.presets()).toHaveLength(1);
  });

  it('apply restores settings and blank geometry', () => {
    targetService.setShape('circle');
    targetService.setDimensions(50, 50);
    sessionTree.patchState((c) => ({
      ...c,
      pipeline: {
        ...c.pipeline,
        settings: { ...c.pipeline.settings, rim_pattern: 'rope' as const },
      },
    }));
    service.saveCurrent('rope-coin');

    // Drift the live state away from the preset.
    targetService.setShape('rectangle');
    targetService.setDimensions(10, 10);
    sessionTree.patchState((c) => ({
      ...c,
      pipeline: {
        ...c.pipeline,
        settings: { ...c.pipeline.settings, rim_pattern: 'none' as const },
      },
    }));

    expect(service.apply('rope-coin')).toBe(true);
    expect(sessionTree.pipeline().settings.rim_pattern).toBe('rope');
    expect(sessionTree.pipeline().settings.zone_width_mm).toBe(50);
    expect(targetService.activeShape()).toBe('circle');
    expect(targetService.printWidthMm()).toBe(50);

    expect(service.apply('does-not-exist')).toBe(false);
  });

  it('remove deletes the preset from memory and storage', () => {
    service.saveCurrent('gone');
    service.remove('gone');
    expect(service.presets()).toHaveLength(0);
    expect(JSON.parse(localStorage.getItem(USER_PRESETS_STORAGE_KEY)!)).toHaveLength(0);
  });

  it('ignores corrupt localStorage payloads', () => {
    localStorage.setItem(USER_PRESETS_STORAGE_KEY, '{not json');
    // Fresh injector so the service re-reads storage.
    TestBed.resetTestingModule();
    TestBed.configureTestingModule({
      providers: [{ provide: ApiClientService, useValue: {} }],
    });
    const fresh = TestBed.inject(SettingsPresetsService);
    expect(fresh.presets()).toEqual([]);
  });
});
