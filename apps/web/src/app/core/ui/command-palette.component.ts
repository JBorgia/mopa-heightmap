import {
  Component,
  HostListener,
  computed,
  inject,
  signal,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';

import { ExportService } from '../state/export.service';
import { MaskService } from '../state/mask.service';
import { PlanService } from '../state/plan.service';
import { RenderService } from '../state/render.service';
import { SculptokService } from '../state/sculptok.service';
import { SessionService } from '../state/session.service';
import { SessionTreeService } from '../state/session-tree.service';

interface PaletteAction {
  id: string;
  label: string;
  group: string;
  shortcut?: string;
  keywords: string;
  execute: () => void;
  disabled?: () => boolean;
}

@Component({
  selector: 'app-command-palette',
  standalone: true,
  imports: [CommonModule],
  template: `
    @if (open()) {
      <div class="palette-backdrop" (click)="close()" aria-hidden="true"></div>
      <div
        class="palette-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Command palette"
      >
        <div class="palette-search-row">
          <span class="palette-icon" aria-hidden="true">⌘</span>
          <input
            #searchInput
            class="palette-input"
            type="text"
            placeholder="Type a command…"
            autocomplete="off"
            spellcheck="false"
            [value]="query()"
            (input)="onQuery($event)"
            (keydown)="onKeydown($event)"
            aria-label="Search commands"
          />
          <kbd class="palette-esc" (click)="close()">Esc</kbd>
        </div>

        <div class="palette-results" role="listbox">
          @if (filtered().length === 0) {
            <div class="palette-empty">No commands match "{{ query() }}"</div>
          }
          @for (group of groupedResults(); track group.name) {
            <div class="palette-group-label">{{ group.name }}</div>
            @for (action of group.actions; track action.id; let i = $index) {
              <button
                type="button"
                class="palette-item"
                [class.palette-item-active]="selectedId() === action.id"
                [disabled]="action.disabled?.() ?? false"
                role="option"
                [attr.aria-selected]="selectedId() === action.id"
                (click)="run(action)"
                (mouseenter)="selectedId.set(action.id)"
              >
                <span class="palette-item-label">{{ action.label }}</span>
                @if (action.shortcut) {
                  <kbd class="palette-shortcut">{{ action.shortcut }}</kbd>
                }
              </button>
            }
          }
        </div>

        <div class="palette-footer">
          <span><kbd>↑↓</kbd> navigate</span>
          <span><kbd>↵</kbd> run</span>
          <span><kbd>Esc</kbd> close</span>
        </div>
      </div>
    }
  `,
  styles: `
    .palette-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.45);
      z-index: 9000;
      backdrop-filter: blur(2px);
    }

    .palette-modal {
      position: fixed;
      top: 12vh;
      left: 50%;
      transform: translateX(-50%);
      width: min(560px, calc(100vw - 2rem));
      z-index: 9001;
      background: var(--bg-surface);
      border: 1px solid var(--border-default);
      border-radius: 1rem;
      box-shadow: 0 24px 48px rgba(0,0,0,0.28), 0 4px 8px rgba(0,0,0,0.12);
      overflow: hidden;
      display: flex;
      flex-direction: column;
      max-height: 70vh;
    }

    .palette-search-row {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.85rem 1rem;
      border-bottom: 1px solid var(--border-default);
    }

    .palette-icon {
      font-size: 1rem;
      color: var(--text-muted);
      flex-shrink: 0;
    }

    .palette-input {
      flex: 1;
      border: none;
      background: transparent;
      font: inherit;
      font-size: 1rem;
      color: var(--text-primary);
      outline: none;
    }

    .palette-input::placeholder {
      color: var(--text-faint);
    }

    .palette-esc {
      flex-shrink: 0;
      font-size: 0.7rem;
      font-family: inherit;
      background: var(--bg-sunken);
      border: 1px solid var(--border-input);
      border-radius: 0.3rem;
      padding: 0.15rem 0.4rem;
      color: var(--text-muted);
      cursor: pointer;
    }

    .palette-results {
      overflow-y: auto;
      flex: 1;
      padding: 0.4rem 0;
    }

    .palette-group-label {
      padding: 0.5rem 1rem 0.2rem;
      font-size: 0.68rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--text-faint);
    }

    .palette-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
      width: 100%;
      padding: 0.55rem 1rem;
      border: none;
      background: transparent;
      font: inherit;
      font-size: 0.9rem;
      color: var(--text-primary);
      text-align: left;
      cursor: pointer;
      border-radius: 0;
    }

    .palette-item:disabled {
      opacity: 0.4;
      cursor: not-allowed;
    }

    .palette-item-active {
      background: color-mix(in srgb, var(--action-bg) 8%, var(--bg-surface));
      color: var(--action-bg);
    }

    .palette-item-label {
      flex: 1;
    }

    .palette-shortcut {
      font-size: 0.7rem;
      font-family: inherit;
      background: var(--bg-sunken);
      border: 1px solid var(--border-input);
      border-radius: 0.3rem;
      padding: 0.1rem 0.4rem;
      color: var(--text-muted);
      white-space: nowrap;
    }

    .palette-empty {
      padding: 2rem 1rem;
      text-align: center;
      color: var(--text-muted);
      font-size: 0.9rem;
    }

    .palette-footer {
      display: flex;
      gap: 1rem;
      padding: 0.5rem 1rem;
      border-top: 1px solid var(--border-default);
      font-size: 0.72rem;
      color: var(--text-faint);
    }

    .palette-footer kbd {
      font-family: inherit;
      background: var(--bg-sunken);
      border: 1px solid var(--border-input);
      border-radius: 0.25rem;
      padding: 0.05rem 0.3rem;
      font-size: 0.68rem;
    }
  `,
})
export class CommandPaletteComponent {
  private readonly router = inject(Router);
  private readonly sessionTree = inject(SessionTreeService);
  private readonly sessionService = inject(SessionService);
  private readonly sculptokService = inject(SculptokService);
  private readonly renderService = inject(RenderService);
  private readonly maskService = inject(MaskService);
  private readonly planService = inject(PlanService);
  private readonly exportService = inject(ExportService);

  protected readonly open = signal(false);
  protected readonly query = signal('');
  protected readonly selectedId = signal<string>('');

  private readonly allActions = computed((): PaletteAction[] => {
    const imageId = this.sessionTree.session()?.imageId;
    const heightmapId = this.sessionTree.output()?.heightmapId;
    const plan = this.sessionTree.output()?.plan;
    const sculptokConfigured = this.sculptokService.credits()?.configured ?? false;

    return [
      // Navigation
      {
        id: 'nav-wizard',
        label: 'Go to Wizard',
        group: 'Navigate',
        keywords: 'wizard navigate page',
        execute: () => this.router.navigate(['/wizard']),
      },
      {
        id: 'nav-studio',
        label: 'Go to Studio',
        group: 'Navigate',
        keywords: 'studio navigate advanced',
        execute: () => this.router.navigate(['/studio']),
      },
      {
        id: 'nav-step-1',
        label: 'Wizard → Step 1: Upload',
        group: 'Navigate',
        keywords: 'upload step 1 photo',
        execute: () => { this.router.navigate(['/wizard']); this.sessionTree.setWizardPage(0); },
      },
      {
        id: 'nav-step-2',
        label: 'Wizard → Step 2: Subject Mask',
        group: 'Navigate',
        keywords: 'mask subject step 2',
        execute: () => { this.router.navigate(['/wizard']); this.sessionTree.setWizardPage(1); },
      },
      {
        id: 'nav-step-3',
        label: 'Wizard → Step 3: Prep & Refine',
        group: 'Navigate',
        keywords: 'prep refine heightmap step 3',
        execute: () => { this.router.navigate(['/wizard']); this.sessionTree.setWizardPage(2); },
      },
      {
        id: 'nav-step-4',
        label: 'Wizard → Step 4: Material & Passes',
        group: 'Navigate',
        keywords: 'material profile passes plan step 4',
        execute: () => { this.router.navigate(['/wizard']); this.sessionTree.setWizardPage(3); },
      },
      {
        id: 'nav-step-5',
        label: 'Wizard → Step 5: Review & Export',
        group: 'Navigate',
        keywords: 'export review download step 5',
        execute: () => { this.router.navigate(['/wizard']); this.sessionTree.setWizardPage(4); },
      },
      // Actions
      {
        id: 'action-sculptok',
        label: 'Generate depth map via Sculptok',
        group: 'Actions',
        keywords: 'sculptok generate depth heightmap ai',
        disabled: () => !imageId || !sculptokConfigured || this.sculptokService.inFlight(),
        execute: () => this.sculptokService.generate(),
      },
      {
        id: 'action-mask',
        label: 'Compute subject mask',
        group: 'Actions',
        keywords: 'mask subject compute birefnet rembg',
        disabled: () => !imageId,
        execute: () => this.maskService.createMask(),
      },
      {
        id: 'action-render',
        label: 'Render heightmap preview',
        group: 'Actions',
        keywords: 'render preview heightmap',
        disabled: () => !imageId || !this.sessionTree.pipeline().settings.external_heightmap_path || this.renderService.inFlight(),
        execute: () => this.renderService.render(),
      },
      {
        id: 'action-plan',
        label: 'Compute pass plan',
        group: 'Actions',
        keywords: 'pass plan compute lightburn',
        disabled: () => !heightmapId || this.planService.inFlight(),
        execute: () => this.planService.computePlan(),
      },
      // Exports
      {
        id: 'export-png',
        label: 'Export PNG (16-bit heightmap)',
        group: 'Export',
        keywords: 'export download png heightmap',
        shortcut: 'PNG',
        disabled: () => !heightmapId,
        execute: () => this.exportService.exportPng(),
      },
      {
        id: 'export-lbrn2',
        label: 'Export .lbrn2 LightBurn project',
        group: 'Export',
        keywords: 'export lightburn lbrn2 project',
        shortcut: '.lbrn2',
        disabled: () => !plan,
        execute: () => this.exportService.exportLbrn2(),
      },
      {
        id: 'export-stl',
        label: 'Export .stl 3D mesh',
        group: 'Export',
        keywords: 'export stl mesh 3d',
        shortcut: '.stl',
        disabled: () => !heightmapId,
        execute: () => this.exportService.exportStl(),
      },
      // Session
      {
        id: 'session-reset',
        label: 'Start a new session (clear all)',
        group: 'Session',
        keywords: 'reset clear new session start over',
        execute: () => {
          if (globalThis.confirm('Clear the current session and start fresh?')) {
            this.sessionTree.reset();
          }
        },
      },
    ];
  });

  protected readonly filtered = computed(() => {
    const q = this.query().toLowerCase().trim();
    if (!q) return this.allActions();
    return this.allActions().filter(
      (a) =>
        a.label.toLowerCase().includes(q) ||
        a.keywords.toLowerCase().includes(q) ||
        a.group.toLowerCase().includes(q),
    );
  });

  protected readonly groupedResults = computed(() => {
    const groups = new Map<string, PaletteAction[]>();
    for (const action of this.filtered()) {
      if (!groups.has(action.group)) groups.set(action.group, []);
      groups.get(action.group)!.push(action);
    }
    return Array.from(groups.entries()).map(([name, actions]) => ({ name, actions }));
  });

  @HostListener('document:keydown', ['$event'])
  onGlobalKeydown(event: KeyboardEvent): void {
    if ((event.metaKey || event.ctrlKey) && event.key === 'k') {
      event.preventDefault();
      this.open() ? this.close() : this.openPalette();
    }
    if (event.key === 'Escape' && this.open()) {
      this.close();
    }
  }

  protected onKeydown(event: KeyboardEvent): void {
    const items = this.filtered();
    if (items.length === 0) return;

    const ids = items.map((a) => a.id);
    const currentIndex = ids.indexOf(this.selectedId());

    if (event.key === 'ArrowDown') {
      event.preventDefault();
      const next = currentIndex < ids.length - 1 ? currentIndex + 1 : 0;
      this.selectedId.set(ids[next]);
    } else if (event.key === 'ArrowUp') {
      event.preventDefault();
      const prev = currentIndex > 0 ? currentIndex - 1 : ids.length - 1;
      this.selectedId.set(ids[prev]);
    } else if (event.key === 'Enter') {
      event.preventDefault();
      const action = items.find((a) => a.id === this.selectedId());
      if (action && !(action.disabled?.() ?? false)) {
        this.run(action);
      }
    }
  }

  protected onQuery(event: Event): void {
    this.query.set((event.target as HTMLInputElement).value);
    const first = this.filtered()[0];
    this.selectedId.set(first?.id ?? '');
  }

  protected run(action: PaletteAction): void {
    this.close();
    action.execute();
  }

  protected close(): void {
    this.open.set(false);
    this.query.set('');
  }

  private openPalette(): void {
    this.open.set(true);
    this.query.set('');
    const first = this.allActions()[0];
    this.selectedId.set(first?.id ?? '');
    // Focus the input after the overlay renders
    globalThis.setTimeout(() => {
      const input = document.querySelector<HTMLInputElement>('.palette-input');
      input?.focus();
    }, 0);
  }
}
