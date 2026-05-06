import { CommonModule } from '@angular/common';
import { Component, computed, inject } from '@angular/core';

import { SharedModule } from 'primeng/api';
import { Card } from 'primeng/card';
import { Splitter } from 'primeng/splitter';

import { ApiClientService } from '../../core/api/api-client.service';
import type { HeightmapSettings } from '../../core/api/api-types';
import { ExportService } from '../../core/state/export.service';
import { MaskService } from '../../core/state/mask.service';
import { PlanService } from '../../core/state/plan.service';
import { RenderService } from '../../core/state/render.service';
import { SessionService } from '../../core/state/session.service';
import { SessionTreeService } from '../../core/state/session-tree.service';
import { MaskBackend, ToastMessage } from '../../core/state/studio-state';

export const WIZARD_PAGE_LABELS = [
  '1. Upload',
  '2. Subject',
  '3. Prep & Refine',
  '4. Material & Passes',
  '5. Review & Export',
] as const;

export const WIZARD_STAGE_SUMMARIES = [
  'Upload a photo of the subject you want to engrave.',
  'Select a mask method and isolate the subject from the background.',
  'Clean the photo before sculptok sees it, and pick refinement layers (subject mask, photo-tonal, signature) to ship in the bundle.',
  'Choose a material profile and preview the engraving pass plan.',
  'Review your settings and export the finished heightmap or pass file.',
] as const;

export const WIZARD_DEFAULT_SPLITTER_SIZES = [68, 32] as const;
export const WIZARD_COLLAPSED_SPLITTER_SIZES = [100, 0] as const;
export const WIZARD_HISTORY_PREVIEW_LIMIT = 5;
export const WIZARD_MASK_BACKENDS: { label: string; value: MaskBackend }[] = [
  { label: 'BiRefNet (best quality)', value: 'birefnet' },
  { label: 'Rembg (fast)', value: 'rembg' },
  { label: 'Threshold (no install needed)', value: 'threshold' },
];

@Component({
  selector: 'app-wizard-shell',
  standalone: true,
  imports: [CommonModule, SharedModule, Card, Splitter],
  template: `
    <!-- Toast notifications -->
    @if (toasts().length > 0) {
      <div class="toast-stack">
        @for (toast of toasts(); track toast.id) {
          <div class="toast" [class]="'toast-' + toast.severity">
            <div class="toast-body">
              <strong>{{ toast.summary }}</strong>
              @if (toast.detail) { <span>{{ toast.detail }}</span> }
            </div>
            <button type="button" class="toast-dismiss" (click)="dismissToast(toast.id)">×</button>
          </div>
        }
      </div>
    }
    <p-splitter
      class="wizard-shell"
      [panelSizes]="splitterSizes()"
      [gutterSize]="8"
      stateKey="mopa-wizard-shell"
      stateStorage="local"
    >
      <ng-template pTemplate="panel">
      <section class="wizard-main">
        <p-card>
          <ng-template pTemplate="header">
            <div class="wizard-header">
              <div>
                <p class="eyebrow">MOPA Heightmap Studio</p>
                <h1>Setup Wizard</h1>
              </div>
              <button type="button" class="secondary" (click)="toggleRightPane()">
                {{ ui().rightPaneCollapsed ? 'Show preview' : 'Hide preview' }}
              </button>
            </div>
            <div class="wizard-page-strip">
              @for (label of wizardPageLabels; track label; let index = $index) {
                <button
                  type="button"
                  class="page-chip"
                  [class.active]="index === ui().wizardPage"
                  (click)="selectPage(index)"
                >
                  {{ label }}
                </button>
              }
            </div>
          </ng-template>

          <div class="wizard-content-grid">
            <section class="wizard-current-page">

              <!-- Page 0: Upload -->
              @if (ui().wizardPage === 0) {
                <h2>{{ wizardPageLabels[0] }}</h2>
                <p>{{ stageSummaries[0] }}</p>
                <div class="wizard-controls">
                  <div class="control-group">
                    <label for="wizard-upload">Source image</label>
                    <input
                      id="wizard-upload"
                      type="file"
                      accept="image/*"
                      (change)="onFileSelected($event)"
                    />
                    <p class="muted">
                      @if (sessionService.uploadInFlight()) {
                        Uploading…
                      } @else if (session().sourceMeta; as meta) {
                        Ready: {{ meta.w }} × {{ meta.h }} px ({{ meta.bytes | number }} bytes)
                      } @else {
                        Upload an image to begin.
                      }
                    </p>
                  </div>
                </div>
              }

              <!-- Page 1: Subject / Mask (BUG-2 fix: clicker key ≠ mask backend) -->
              @if (ui().wizardPage === 1) {
                <h2>{{ wizardPageLabels[1] }}</h2>
                <p>{{ stageSummaries[1] }}</p>
                <div class="wizard-controls">
                  <div class="control-group">
                    <label for="wiz-mask-backend">Mask backend</label>
                    <select
                      id="wiz-mask-backend"
                      [value]="pipeline().mask.backend"
                      (change)="onMaskBackendChange($event)"
                    >
                      @for (opt of maskBackends; track opt.value) {
                        <option [value]="opt.value" [selected]="opt.value === pipeline().mask.backend">{{ opt.label }}</option>
                      }
                    </select>
                  </div>
                  <div class="control-group">
                    <label for="wiz-edge-softness">
                      Edge softness: {{ pipeline().mask.edgeSoftness }}
                    </label>
                    <input
                      id="wiz-edge-softness"
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      [value]="pipeline().mask.edgeSoftness"
                      (change)="onEdgeSoftnessChange($event)"
                    />
                  </div>
                  <div class="control-actions">
                    <button type="button" [disabled]="!session().imageId" (click)="createMask()">
                      Compute mask
                    </button>
                  </div>
                  @if (pipeline().mask.maskId) {
                    <p class="muted">
                      Mask computed — coverage {{ pipeline().mask.coveragePct | number: '1.1-1' }}%
                    </p>
                    <div class="click-refine-panel">
                      <p class="refine-hint">
                        Click on the image preview (right pane) to refine the mask boundary
                        using flood-fill. The clicker uses its own registry key — not the
                        mask backend.
                      </p>
                      <div class="control-actions">
                        <button
                          type="button"
                          class="secondary"
                          (click)="clickRefineAt(0.5, 0.5, 'positive')"
                        >
                          + Include centre
                        </button>
                        <button
                          type="button"
                          class="secondary"
                          (click)="clickRefineAt(0.05, 0.05, 'negative')"
                        >
                          – Exclude corner
                        </button>
                      </div>
                    </div>
                  }
                </div>
              }

              <!-- Page 2: Prep & Refine -->
              @if (ui().wizardPage === 2) {
                <h2>{{ wizardPageLabels[2] }}</h2>
                <p>{{ stageSummaries[2] }}</p>
                <div class="wizard-controls">

                  <fieldset class="control-section">
                    <legend>Pre-sculptok prep</legend>
                    <label class="control-toggle">
                      <input type="checkbox"
                        [checked]="pipeline().settings.input_clahe"
                        (change)="onSettingToggle('input_clahe', $event)" />
                      CLAHE contrast
                    </label>
                    <label class="control-toggle">
                      <input type="checkbox"
                        [checked]="pipeline().settings.input_denoise"
                        (change)="onSettingToggle('input_denoise', $event)" />
                      Denoise
                    </label>
                    <label class="control-toggle">
                      <input type="checkbox"
                        [checked]="pipeline().settings.input_remove_specular"
                        (change)="onSettingToggle('input_remove_specular', $event)" />
                      Remove specular highlights
                    </label>
                  </fieldset>

                  <fieldset class="control-section">
                    <legend>Refinement passes</legend>
                    <label class="control-toggle">
                      <input type="checkbox"
                        [checked]="pipeline().settings.subject_mask_enabled"
                        (change)="onSettingToggle('subject_mask_enabled', $event)" />
                      Subject mask deliverable
                    </label>
                    <label class="control-toggle">
                      <input type="checkbox"
                        [checked]="pipeline().settings.pre_clean_enabled"
                        (change)="onSettingToggle('pre_clean_enabled', $event)" />
                      Pre-clean pass
                    </label>
                    <label class="control-toggle">
                      <input type="checkbox"
                        [checked]="pipeline().settings.photo_tonal_enabled"
                        (change)="onSettingToggle('photo_tonal_enabled', $event)" />
                      Photo-tonal overlay
                    </label>
                    <label class="control-toggle">
                      <input type="checkbox"
                        [checked]="pipeline().settings.polarity_invert"
                        (change)="onSettingToggle('polarity_invert', $event)" />
                      Polarity invert (signet ring)
                    </label>
                    <div class="control-group">
                      <label for="wiz-sig-text">Signature text</label>
                      <input id="wiz-sig-text" type="text" maxlength="64"
                        placeholder="e.g. JB 2026"
                        [value]="pipeline().settings.signature_text"
                        (change)="onSettingValue('signature_text', $event)" />
                    </div>
                  </fieldset>

                  <div class="control-actions">
                    <button type="button"
                      [disabled]="!canRender()"
                      (click)="renderPreview()">
                      Render preview
                    </button>
                  </div>
                  @if (!pipeline().settings.external_heightmap_path && session().imageId) {
                    <p class="muted">
                      Render needs a heightmap source. Generate one in the Studio's
                      Heightmap panel (Generate via Sculptok) or supply your own PNG.
                    </p>
                  }
                  @if (output().elapsedSeconds !== null) {
                    <p class="muted">Rendered in {{ output().elapsedSeconds | number: '1.2-2' }} s</p>
                  }
                </div>
              }

              <!-- Page 3: Material & Passes -->
              @if (ui().wizardPage === 3) {
                <h2>{{ wizardPageLabels[3] }}</h2>
                <p>{{ stageSummaries[3] }}</p>
                <div class="wizard-controls">
                  <div class="control-group">
                    <label for="wiz-profile">Material profile</label>
                    <select
                      id="wiz-profile"
                      [value]="pipeline().render.profileName ?? ''"
                      (change)="onProfileSelected($event)"
                    >
                      <option value="" [selected]="!pipeline().render.profileName">Select a profile…</option>
                      @for (profile of sessionService.profiles(); track profile.name) {
                        <option [value]="profile.name" [selected]="profile.name === pipeline().render.profileName">{{ profile.name }}</option>
                      }
                    </select>
                  </div>
                  <div class="control-actions">
                    <button
                      type="button"
                      [disabled]="!output().heightmapId"
                      (click)="computePlan()"
                    >
                      Compute pass plan
                    </button>
                  </div>
                  @if (output().plan; as plan) {
                    <section class="plan-summary">
                      <h3>Pass plan ({{ plan.passes.length }} passes)</h3>
                      <ol class="pass-list">
                        @for (p of plan.passes; track p.passNumber) {
                          <li>
                            <span class="swatch" [style.background]="p.colorHex"></span>
                            {{ p.label }}
                          </li>
                        }
                      </ol>
                    </section>
                  }
                </div>
              }

              <!-- Page 4: Review & Export -->
              @if (ui().wizardPage === 4) {
                <h2>{{ wizardPageLabels[4] }}</h2>
                <p>{{ stageSummaries[4] }}</p>
                <div class="wizard-controls">
                  <dl class="summary-grid">
                    <div>
                      <dt>Image</dt>
                      <dd>{{ session().imageId ? 'Uploaded ✓' : 'Not uploaded' }}</dd>
                    </div>
                    <div>
                      <dt>Mask</dt>
                      <dd>{{ pipeline().mask.maskId ? 'Ready — ' + (pipeline().mask.coveragePct | number: '1.1-1') + '% coverage' : 'Not computed' }}</dd>
                    </div>
                    <div>
                      <dt>Heightmap</dt>
                      <dd>{{ output().heightmapId ? 'Rendered ✓' : 'Not rendered' }}</dd>
                    </div>
                    <div>
                      <dt>Pass plan</dt>
                      <dd>{{ output().plan ? (output().plan!.passes.length + ' passes ready') : 'Not computed' }}</dd>
                    </div>
                  </dl>
                  <div class="control-actions">
                    <button
                      type="button"
                      class="secondary"
                      [disabled]="!output().heightmapId"
                      (click)="exportPng()"
                    >
                      Export PNG
                    </button>
                    <button
                      type="button"
                      class="secondary"
                      [disabled]="!output().plan"
                      (click)="exportLbrn2()"
                    >
                      Export .lbrn2
                    </button>
                    <button
                      type="button"
                      class="secondary"
                      [disabled]="!output().heightmapId"
                      (click)="exportStl()"
                    >
                      Export .stl
                    </button>
                  </div>
                </div>
              }

              <div class="wizard-actions">
                <button type="button" class="secondary" (click)="previousPage()" [disabled]="ui().wizardPage === 0">
                  ← Previous
                </button>
                <button type="button" (click)="nextPage()" [disabled]="ui().wizardPage === wizardPageLabels.length - 1">
                  Next →
                </button>
              </div>
            </section>

            <section class="wizard-history">
              <h2>Session activity</h2>
              @if (historyPreview().length > 0) {
                <ul>
                  @for (entry of historyPreview(); track entry.id) {
                    <li>
                      <strong>{{ entry.action }}</strong>
                      <span>
                        {{ relativeTime(entry.timestampIso) }}
                        @if (entry.durationMs !== undefined) {
                          · {{ formatDuration(entry.durationMs) }}
                        }
                      </span>
                    </li>
                  }
                </ul>
              } @else {
                <p class="muted">No actions recorded yet.</p>
              }
            </section>
          </div>
        </p-card>
      </section>
      </ng-template>

      <ng-template pTemplate="panel">
      <aside class="wizard-sidepane">
        <p-card header="Preview" subheader="Session state">
          <div class="preview-tile">
            <h3>Source</h3>
            @if (session().sourceMeta; as meta) {
              <p>{{ meta.w }} × {{ meta.h }} px</p>
            } @else {
              <p class="muted">No image uploaded yet.</p>
            }
          </div>

          <div class="preview-tile">
            <h3>Heightmap</h3>
            @if (output().previewId; as pid) {
              <img [src]="blobUrl(pid)" alt="Heightmap preview" style="width:100%;" />
            } @else {
              <p class="muted">Preview will appear after rendering.</p>
            }
          </div>

          <dl class="state-grid compact">
            <div>
              <dt>Hash</dt>
              <dd>{{ session().imageHash ? (session().imageHash!.slice(0, 8) + '…') : '—' }}</dd>
            </div>
            <div>
              <dt>Coverage</dt>
              <dd>{{ pipeline().mask.coveragePct | number: '1.1-1' }}%</dd>
            </div>
            <div>
              <dt>Elapsed</dt>
              <dd>{{ output().elapsedSeconds !== null ? ((output().elapsedSeconds! | number: '1.2-2') + ' s') : '—' }}</dd>
            </div>
            <div>
              <dt>Passes</dt>
              <dd>{{ output().plan ? output().plan!.passes.length : '—' }}</dd>
            </div>
          </dl>
        </p-card>
      </aside>
      </ng-template>
    </p-splitter>
  `,
  styles: `
    :host {
      display: block;
      min-height: calc(100vh - 5rem);
    }

    .wizard-shell {
      min-height: calc(100vh - 5rem);
    }

    .wizard-main,
    .wizard-sidepane {
      height: 100%;
      padding: 0.75rem;
    }

    .wizard-header {
      display: flex;
      justify-content: space-between;
      gap: 1rem;
      align-items: flex-start;
      padding: 1.5rem 1.5rem 0;
    }

    .eyebrow {
      margin: 0 0 0.5rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 0.75rem;
      color: var(--text-muted);
    }

    h1,
    h2,
    h3,
    p {
      margin-top: 0;
    }

    .wizard-page-strip {
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 0.4rem;
      padding: 0.75rem 1.5rem 0;
    }

    .page-chip,
    button {
      border-radius: 999px;
      border: 1px solid var(--border-input);
      background: var(--bg-surface);
      color: var(--text-primary);
      padding: 0.45rem 0.5rem;
      font: inherit;
      font-size: 0.8rem;
      line-height: 1.25;
      text-align: center;
      cursor: pointer;
    }

    .page-chip.active,
    button:not(.secondary):not(.page-chip):not(.toast-dismiss) {
      background: var(--action-bg);
      color: var(--action-fg);
      border-color: var(--action-bg);
    }

    button.secondary {
      background: var(--action-secondary-bg);
      color: var(--action-secondary-fg);
      border-color: var(--border-input);
    }

    button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }

    .wizard-content-grid {
      display: grid;
      grid-template-columns: minmax(0, 2fr) minmax(18rem, 1fr);
      gap: 1rem;
      padding: 0 1.5rem 1.5rem;
    }

    .wizard-current-page,
    .wizard-history {
      border: 1px solid var(--border-default);
      border-radius: 1rem;
      padding: 1rem;
      background: var(--bg-sunken);
    }

    .preview-tile {
      border: 1px solid var(--border-default);
      border-radius: 0;
      padding: 1rem;
      background: var(--bg-sunken);
    }

    .wizard-actions {
      display: flex;
      justify-content: space-between;
      gap: 0.75rem;
      margin-top: 1.25rem;
    }

    .wizard-controls {
      display: grid;
      gap: 1rem;
      margin-top: 1.25rem;
      padding: 1rem;
      border: 1px solid var(--border-default);
      border-radius: 1rem;
      background: var(--bg-surface);
    }

    .control-group {
      display: grid;
      gap: 0.45rem;
    }

    .control-group label {
      font-size: 0.9rem;
      font-weight: 600;
      color: var(--text-primary);
    }

    .control-group input[type=text],
    .control-group input[type=file],
    .control-group select {
      border: 1px solid var(--border-input);
      border-radius: 0.75rem;
      background: var(--bg-input);
      color: var(--text-primary);
      padding: 0.7rem 0.85rem;
      font: inherit;
    }

    .control-group input[type=range] {
      width: 100%;
      accent-color: var(--action-bg);
    }

    .control-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
    }

    .state-grid,
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.75rem;
      margin: 1rem 0 0;
    }

    .state-grid.compact {
      margin-top: 1.25rem;
    }

    .state-grid.compact dt {
      font-size: 0.68rem;
      letter-spacing: 0.05em;
    }

    .state-grid div,
    .summary-grid div {
      border: 1px solid var(--border-default);
      border-radius: 0.75rem;
      background: var(--bg-surface);
      padding: 0.75rem;
    }

    dt {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-muted);
      margin-bottom: 0.35rem;
    }

    dd {
      margin: 0;
      font-weight: 600;
      color: var(--text-primary);
      overflow-wrap: anywhere;
    }

    .plan-summary {
      border-top: 1px solid var(--border-default);
      padding-top: 1rem;
    }

    .pass-list {
      padding-left: 0;
      list-style: none;
      display: grid;
      gap: 0.5rem;
    }

    .pass-list li {
      display: flex;
      align-items: center;
      gap: 0.6rem;
      border-bottom: none;
    }

    .swatch {
      display: inline-block;
      width: 1rem;
      height: 1rem;
      border-radius: 0.25rem;
      border: 1px solid var(--border-input);
      flex-shrink: 0;
    }

    .click-refine-panel {
      border-top: 1px solid var(--border-default);
      padding-top: 0.75rem;
    }

    .refine-hint {
      font-size: 0.85rem;
      color: var(--text-muted);
    }

    ul {
      list-style: none;
      padding: 0;
      margin: 0;
      display: grid;
      gap: 0.75rem;
    }

    li {
      display: flex;
      justify-content: space-between;
      gap: 0.75rem;
      border-bottom: 1px solid var(--border-default);
      padding-bottom: 0.75rem;
    }

    li:last-child {
      border-bottom: 0;
      padding-bottom: 0;
    }

    .muted {
      color: var(--text-muted);
    }

    .toast-stack {
      position: fixed;
      top: 1rem;
      right: 1rem;
      z-index: 9999;
      display: grid;
      gap: 0.5rem;
      width: min(28rem, calc(100vw - 2rem));
    }

    .toast {
      display: flex;
      align-items: flex-start;
      gap: 0.75rem;
      border-radius: 0.75rem;
      padding: 0.75rem 1rem;
      border: 1px solid var(--border-default);
      background: var(--bg-surface);
      box-shadow: 0 4px 16px rgba(0,0,0,0.18);
    }

    .toast-error   { border-color: #e74c3c; background: color-mix(in srgb, #e74c3c 12%, var(--bg-surface)); }
    .toast-warn    { border-color: #e67e22; background: color-mix(in srgb, #e67e22 12%, var(--bg-surface)); }
    .toast-success { border-color: #27ae60; background: color-mix(in srgb, #27ae60 12%, var(--bg-surface)); }
    .toast-info    { border-color: var(--action-bg); background: color-mix(in srgb, var(--action-bg) 12%, var(--bg-surface)); }

    .toast-body {
      flex: 1;
      display: grid;
      gap: 0.2rem;
      font-size: 0.875rem;
    }

    .toast-body strong { color: var(--text-primary); }
    .toast-body span   { color: var(--text-secondary); word-break: break-word; }

    .toast-dismiss {
      border: none;
      background: transparent;
      cursor: pointer;
      color: var(--text-muted);
      font-size: 1.2rem;
      line-height: 1;
      padding: 0;
      border-radius: 0;
      flex-shrink: 0;
      align-self: flex-start;
    }

    @media (max-width: 960px) {
      .wizard-content-grid {
        grid-template-columns: 1fr;
      }
    }
  `,
})
export class WizardShellComponent {
  protected readonly wizardPageLabels = WIZARD_PAGE_LABELS;
  protected readonly stageSummaries = WIZARD_STAGE_SUMMARIES;
  protected readonly maskBackends = WIZARD_MASK_BACKENDS;
  protected readonly sessionTree = inject(SessionTreeService);
  protected readonly sessionService = inject(SessionService);
  private readonly apiClient = inject(ApiClientService);
  private readonly maskService = inject(MaskService);
  private readonly renderService = inject(RenderService);
  private readonly planService = inject(PlanService);
  private readonly exportService = inject(ExportService);
  protected readonly session = this.sessionTree.session;
  protected readonly pipeline = this.sessionTree.pipeline;
  protected readonly output = this.sessionTree.output;
  protected readonly ui = this.sessionTree.ui;
  protected readonly toasts = computed(() => this.ui().toasts);
  protected readonly splitterSizes = computed(() => this.ui().rightPaneCollapsed
    ? [...WIZARD_COLLAPSED_SPLITTER_SIZES]
    : [...WIZARD_DEFAULT_SPLITTER_SIZES]);
  protected readonly historyPreview = computed(() => this.session().history.slice(0, WIZARD_HISTORY_PREVIEW_LIMIT));

  constructor() {
    this.sessionService.loadProfiles();
  }

  protected blobUrl(id: string): string {
    return this.apiClient.blobUrl(id);
  }

  protected relativeTime(iso: string | null | undefined): string {
    if (!iso) return '—';
    const t = new Date(iso).getTime();
    if (Number.isNaN(t)) return '—';
    const seconds = Math.max(0, Math.round((Date.now() - t) / 1000));
    if (seconds < 5) return 'just now';
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.round(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.round(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.round(hours / 24);
    return `${days}d ago`;
  }

  protected formatDuration(ms: number | null | undefined): string {
    if (ms === null || ms === undefined || Number.isNaN(ms)) return '';
    if (ms < 1000) return `${Math.round(ms)} ms`;
    const s = ms / 1000;
    if (s < 60) return `${s.toFixed(1)} s`;
    const totalSec = Math.round(s);
    const minutes = Math.floor(totalSec / 60);
    const seconds = totalSec % 60;
    return `${minutes}m ${seconds.toString().padStart(2, '0')}s`;
  }

  protected onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.item(0);
    if (!file) return;
    this.sessionService.uploadImage(file);
    input.value = '';
  }

  protected onProfileSelected(event: Event): void {
    const select = event.target as HTMLSelectElement;
    this.sessionService.setProfileName(select.value || null);
  }

  protected onMaskBackendChange(event: Event): void {
    const select = event.target as HTMLSelectElement;
    this.maskService.setBackend(select.value as MaskBackend);
  }

  protected onEdgeSoftnessChange(event: Event): void {
    const input = event.target as HTMLInputElement;
    this.maskService.setEdgeSoftness(parseFloat(input.value));
  }

  protected createMask(): void {
    this.maskService.createMask();
  }

  /** Generic toggle handler for boolean ``HeightmapSettings`` keys. */
  protected onSettingToggle<K extends keyof HeightmapSettings>(key: K, event: Event): void {
    const checked = (event.target as HTMLInputElement).checked;
    this.renderService.patchSettings(key, checked as HeightmapSettings[K]);
  }

  /** Generic string-or-enum handler (selects, text inputs). */
  protected onSettingValue<K extends keyof HeightmapSettings>(key: K, event: Event): void {
    const value = (event.target as HTMLInputElement | HTMLSelectElement).value;
    this.renderService.patchSettings(key, value as HeightmapSettings[K]);
  }

  /**
   * BUG-2 fix: click-refine dispatches `clickRefine()` on MaskService which
   * uses `clicker_key` from state — never the mask backend dropdown value.
   * x/y are normalised [0,1] fractions of the image dimensions.
   */
  protected clickRefineAt(x: number, y: number, label: 'positive' | 'negative'): void {
    const meta = this.session().sourceMeta;
    if (!meta) return;
    this.maskService.clickRefine(Math.round(x * meta.w), Math.round(y * meta.h), label);
  }

  protected renderPreview(): void {
    this.renderService.render();
  }

  /** Same gate as the Studio: image uploaded + heightmap source set. */
  protected canRender(): boolean {
    const state = this.sessionTree.state();
    return Boolean(
      state.session.imageId && state.pipeline.settings.external_heightmap_path,
    );
  }

  protected computePlan(): void {
    this.planService.computePlan();
  }

  protected exportPng(): void {
    this.exportService.exportPng();
  }

  protected exportLbrn2(): void {
    this.exportService.exportLbrn2();
  }

  protected exportStl(): void {
    this.exportService.exportStl();
  }

  protected dismissToast(id: string): void {
    this.sessionTree.clearToast(id);
  }

  protected selectPage(index: number): void {
    const page = index as 0 | 1 | 2 | 3 | 4;
    this.sessionTree.setWizardPage(page);
    this.sessionTree.pushHistory(`Navigated to ${this.wizardPageLabels[page]}`);
  }

  protected previousPage(): void {
    const previous = Math.max(0, this.ui().wizardPage - 1) as 0 | 1 | 2 | 3 | 4;
    this.selectPage(previous);
  }

  protected nextPage(): void {
    const next = Math.min(this.wizardPageLabels.length - 1, this.ui().wizardPage + 1) as 0 | 1 | 2 | 3 | 4;
    this.selectPage(next);
  }

  protected toggleRightPane(): void {
    const collapsed = !this.ui().rightPaneCollapsed;
    this.sessionTree.setRightPaneCollapsed(collapsed);
    this.sessionTree.pushHistory(collapsed ? 'Collapsed wizard right pane' : 'Expanded wizard right pane');
  }
}

