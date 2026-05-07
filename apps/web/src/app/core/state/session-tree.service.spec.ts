import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { describe, it, expect, beforeEach, vi } from 'vitest';

import { SessionTreeService } from './session-tree.service';
import { DEFAULT_STUDIO_STATE } from './studio-state';

describe('SessionTreeService', () => {
  let service: SessionTreeService;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideRouter([])],
    });
    service = TestBed.inject(SessionTreeService);
    service.reset();
  });

  it('initial state matches DEFAULT_STUDIO_STATE structure', () => {
    const state = service.state();
    expect(state.session.imageId).toBe(DEFAULT_STUDIO_STATE.session.imageId);
    expect(state.ui.wizardPage).toBe(DEFAULT_STUDIO_STATE.ui.wizardPage);
  });

  it('reset() restores defaults', () => {
    service.setWizardPage(3);
    service.reset();
    expect(service.state().ui.wizardPage).toBe(DEFAULT_STUDIO_STATE.ui.wizardPage);
  });

  it('setSessionImage() patches session fields', () => {
    service.setSessionImage('img-x', 'hash-x', null);
    const { session } = service.state();
    expect(session.imageId).toBe('img-x');
    expect(session.imageHash).toBe('hash-x');
    expect(session.sourceMeta).toBeNull();
  });

  it('pushHistory() prepends an entry', () => {
    service.pushHistory('test:action');
    const { history } = service.state().session;
    expect(history[0].action).toBe('test:action');
    expect(history[0].id).toBeTruthy();
    expect(history[0].timestampIso).toBeTruthy();
  });

  it('pushHistory() trims history to STUDIO_HISTORY_LIMIT', async () => {
    const { STUDIO_HISTORY_LIMIT } = await import('./studio-state');
    for (let i = 0; i < STUDIO_HISTORY_LIMIT + 5; i++) {
      service.pushHistory(`action:${i}`);
    }
    expect(service.state().session.history.length).toBe(STUDIO_HISTORY_LIMIT);
  });

  it('setActiveRoute() updates ui.activeRoute', () => {
    service.setActiveRoute('studio');
    expect(service.state().ui.activeRoute).toBe('studio');
  });

  it('setWizardPage() updates ui.wizardPage', () => {
    service.setWizardPage(2);
    expect(service.state().ui.wizardPage).toBe(2);
  });

  it('setRightPaneCollapsed() updates ui.rightPaneCollapsed', () => {
    service.setRightPaneCollapsed(true);
    expect(service.state().ui.rightPaneCollapsed).toBe(true);
  });

  it('addToast() appends a toast entry', () => {
    service.addToast({ id: 't1', severity: 'info', summary: 'Test', detail: '' });
    expect(service.state().ui.toasts[0].id).toBe('t1');
  });

  it('toasts are stripped on deserialize so they do not haunt page reloads', async () => {
    const { deserializeStudioState, DEFAULT_STUDIO_STATE } = await import('./studio-state');
    const persisted = JSON.stringify({
      ...DEFAULT_STUDIO_STATE,
      ui: {
        ...DEFAULT_STUDIO_STATE.ui,
        toasts: [
          { id: 'persisted-error', severity: 'error', summary: 'Hi', detail: '' },
        ],
      },
    });
    const restored = deserializeStudioState(persisted);
    expect(restored.ui.toasts).toEqual([]);
  });

  it('clearToast() removes the matching toast', () => {
    service.addToast({ id: 't2', severity: 'warn', summary: 'W', detail: '' });
    service.clearToast('t2');
    expect(service.state().ui.toasts.find((t) => t.id === 't2')).toBeUndefined();
  });

  it('patchState() applies arbitrary updater', () => {
    service.patchState((current) => ({
      ...current,
      ui: { ...current.ui, wizardPage: 4 as (typeof current.ui)['wizardPage'] },
    }));
    expect(service.state().ui.wizardPage).toBe(4);
  });

  it('session signal reflects current session', () => {
    service.setSessionImage('img-y', 'hash-y', null);
    expect(service.session().imageId).toBe('img-y');
  });

  it('pipeline signal reflects current pipeline', () => {
    expect(service.pipeline()).toBeDefined();
  });

  it('output signal reflects current output', () => {
    expect(service.output()).toBeDefined();
  });

  it('ui signal reflects current ui', () => {
    service.setWizardPage(1);
    expect(service.ui().wizardPage).toBe(1);
  });
});
