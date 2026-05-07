import { Injectable, effect, inject } from '@angular/core';
import { NavigationEnd, Router } from '@angular/router';
import { filter } from 'rxjs';

import { createSignalTreeStore } from '../../../lib/signal-tree';
import {
  ActiveRoute,
  HistoryEntry,
  LOCAL_STORAGE_DEBOUNCE_MS,
  STUDIO_HISTORY_LIMIT,
  STUDIO_STATE_STORAGE_KEY,
  StudioState,
  cloneDefaultStudioState,
  deserializeStudioState,
  serializeStudioState,
} from './studio-state';

export const TOAST_AUTO_DISMISS_MS = 5000;

@Injectable({ providedIn: 'root' })
export class SessionTreeService {
  private readonly router = inject(Router);
  private readonly store = createSignalTreeStore<StudioState>(
    deserializeStudioState(globalThis.localStorage?.getItem(STUDIO_STATE_STORAGE_KEY) ?? null),
  );
  private persistTimerId: number | null = null;

  readonly state = this.store.state.asReadonly();
  readonly session = this.store.select((state) => state.session);
  readonly pipeline = this.store.select((state) => state.pipeline);
  readonly output = this.store.select((state) => state.output);
  readonly ui = this.store.select((state) => state.ui);

  constructor() {
    effect(() => {
      const snapshot = this.state();
      this.schedulePersist(snapshot);
    });

    this.router.events.pipe(filter((event) => event instanceof NavigationEnd)).subscribe((event) => {
      const navigation = event as NavigationEnd;
      this.setActiveRoute(this.routeFromUrl(navigation.urlAfterRedirects));
    });
  }

  reset(): void {
    this.store.state.set(cloneDefaultStudioState());
  }

  setSessionImage(imageId: string, imageHash: string, sourceMeta: StudioState['session']['sourceMeta']): void {
    this.store.patch((current) => ({
      ...current,
      session: {
        ...current.session,
        imageId,
        imageHash,
        sourceMeta,
      },
    }));
  }

  pushHistory(action: string, durationMs?: number): void {
    const historyEntry: HistoryEntry = {
      id: crypto.randomUUID(),
      action,
      timestampIso: new Date().toISOString(),
      ...(durationMs !== undefined ? { durationMs } : {}),
    };

    this.store.patch((current) => ({
      ...current,
      session: {
        ...current.session,
        history: [historyEntry, ...current.session.history].slice(0, STUDIO_HISTORY_LIMIT),
      },
    }));
  }

  setActiveRoute(route: ActiveRoute): void {
    this.store.patch((current) => ({
      ...current,
      ui: {
        ...current.ui,
        activeRoute: route,
      },
    }));
  }

  setWizardPage(page: StudioState['ui']['wizardPage']): void {
    this.store.patch((current) => ({
      ...current,
      ui: {
        ...current.ui,
        wizardPage: page,
      },
    }));
  }

  setRightPaneCollapsed(rightPaneCollapsed: boolean): void {
    this.store.patch((current) => ({
      ...current,
      ui: {
        ...current.ui,
        rightPaneCollapsed,
      },
    }));
  }

  addToast(toast: StudioState['ui']['toasts'][number]): void {
    this.store.patch((current) => ({
      ...current,
      ui: {
        ...current.ui,
        toasts: [...current.ui.toasts, toast],
      },
    }));
    // Auto-dismiss success/info toasts so they don't pile up. Errors and
    // warnings stay until the user dismisses them — they signal something
    // that needs attention.
    if (toast.severity === 'success' || toast.severity === 'info') {
      globalThis.setTimeout(() => this.clearToast(toast.id), TOAST_AUTO_DISMISS_MS);
    }
  }

  clearToast(toastId: string): void {
    this.store.patch((current) => ({
      ...current,
      ui: {
        ...current.ui,
        toasts: current.ui.toasts.filter((toast) => toast.id !== toastId),
      },
    }));
  }

  patchState(updater: (current: StudioState) => StudioState): void {
    this.store.patch(updater);
  }

  private routeFromUrl(url: string): ActiveRoute {
    if (url.startsWith('/studio')) {
      return 'studio';
    }
    if (url.startsWith('/export')) {
      return 'export';
    }
    return 'wizard';
  }

  private schedulePersist(state: StudioState): void {
    if (this.persistTimerId !== null) {
      clearTimeout(this.persistTimerId);
    }

    this.persistTimerId = globalThis.setTimeout(() => {
      globalThis.localStorage?.setItem(STUDIO_STATE_STORAGE_KEY, serializeStudioState(state));
      this.persistTimerId = null;
    }, LOCAL_STORAGE_DEBOUNCE_MS);
  }
}