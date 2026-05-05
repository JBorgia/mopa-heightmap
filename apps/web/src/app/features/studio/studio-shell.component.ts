import { CommonModule } from '@angular/common';
import { Component, computed, inject } from '@angular/core';

import { Accordion, AccordionContent, AccordionHeader, AccordionPanel } from 'primeng/accordion';
import { Card } from 'primeng/card';

import { ApiClientService } from '../../core/api/api-client.service';
import { ExportService } from '../../core/state/export.service';
import { MaskService } from '../../core/state/mask.service';
import { PlanService } from '../../core/state/plan.service';
import { RenderService } from '../../core/state/render.service';
import { SessionService } from '../../core/state/session.service';
import { SessionTreeService } from '../../core/state/session-tree.service';
import { MaskBackend } from '../../core/state/studio-state';

export const STUDIO_SECTION_TITLES = {
  mask: 'Mask',
  render: 'Render',
  advanced: 'Advanced',
  output: 'Output',
} as const;

export const STUDIO_MASK_BACKENDS: { label: string; value: MaskBackend }[] = [
  { label: 'BiRefNet (best quality)', value: 'birefnet' },
  { label: 'RemBG (fast)', value: 'rembg' },
  { label: 'Threshold (no install needed)', value: 'threshold' },
];

export const STUDIO_UPSCALER_OPTIONS = [
  { label: 'Real-ESRGAN', value: 'realesrgan' },
  { label: 'SwinIR', value: 'swinir' },
] as const;

@Component({
  selector: 'app-studio-shell',
  standalone: true,
  imports: [CommonModule, Card, Accordion, AccordionPanel, AccordionHeader, AccordionContent],
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
    <div class="studio-layout">
      <header class="studio-topbar">
        <p class="eyebrow">MOPA Heightmap Studio</p>
        <h1>Studio</h1>
        <div class="topbar-meta">
          @if (session().sourceMeta; as meta) {
            <span class="badge">{{ meta.w }} × {{ meta.h }} px</span>
          } @else {
            <span class="badge muted">No image loaded</span>
          }
          @if (output().heightmapId) {
            <span class="badge success">Heightmap ready</span>
          }
        </div>
      </header>

      <div class="studio-body">
        <!-- Left: accordion controls -->
        <main class="studio-controls">
          <!-- Upload strip (always visible) -->
          <div class="upload-strip">
            <label for="studio-upload" class="upload-label">Source image</label>
            <input
              id="studio-upload"
              type="file"
              accept="image/*"
              (change)="onFileSelected($event)"
            />
            @if (sessionService.uploadInFlight()) {
              <span class="hint">Uploading…</span>
            }
          </div>

          <p-accordion [multiple]="true" [value]="['mask', 'render']">
            <!-- MASK panel -->
            <p-accordion-panel value="mask">
              <p-accordion-header>{{ sectionTitles.mask }}</p-accordion-header>
              <p-accordion-content>
                <div class="panel-body">
                  <div class="field">
                    <label for="mask-backend">Backend</label>
                    <select
                      id="mask-backend"
                      [value]="pipeline().mask.backend"
                      (change)="onMaskBackendChange($event)"
                    >
                      @for (opt of maskBackends; track opt.value) {
                        <option [value]="opt.value" [selected]="opt.value === pipeline().mask.backend">{{ opt.label }}</option>
                      }
                    </select>
                  </div>
                  <div class="field">
                    <label>Edge softness <span class="value-badge">{{ pipeline().mask.edgeSoftness | number:'1.2-2' }}</span></label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      [value]="pipeline().mask.edgeSoftness"
                      (change)="onEdgeSoftnessChange($event)"
                    />
                  </div>
                  @if (pipeline().mask.maskId) {
                    <p class="hint">Coverage: {{ pipeline().mask.coveragePct | number:'1.1-1' }}%</p>
                  }
                  <div class="actions">
                    <button type="button" [disabled]="!session().imageId" (click)="createMask()">
                      Compute mask
                    </button>
                  </div>
                </div>
              </p-accordion-content>
            </p-accordion-panel>

            <!-- RENDER panel -->
            <p-accordion-panel value="render">
              <p-accordion-header>{{ sectionTitles.render }}</p-accordion-header>
              <p-accordion-content>
                <div class="panel-body">
                  <div class="field">
                    <label>Detail balance <span class="value-badge">{{ pipeline().render.detailBalance | number:'1.2-2' }}</span></label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      [value]="pipeline().render.detailBalance"
                      (change)="onDetailBalanceChange($event)"
                    />
                  </div>
                  <div class="field">
                    <label>Relief strength <span class="value-badge">{{ pipeline().render.relief | number:'1.2-2' }}</span></label>
                    <input
                      type="range"
                      min="0"
                      max="2"
                      step="0.01"
                      [value]="pipeline().render.relief"
                      (change)="onReliefChange($event)"
                    />
                  </div>
                  <div class="field field-toggle">
                    <label for="multires">Multi-resolution</label>
                    <input
                      id="multires"
                      type="checkbox"
                      [checked]="pipeline().render.multires"
                      (change)="onMultiresChange($event)"
                    />
                  </div>
                  <div class="field">
                    <label for="render-profile">Material profile</label>
                    <select
                      id="render-profile"
                      [value]="pipeline().render.profileName ?? ''"
                      (change)="onProfileChange($event)"
                    >
                      <option value="" [selected]="!pipeline().render.profileName">— none —</option>
                      @for (p of sessionService.profiles(); track p.name) {
                        <option [value]="p.name" [selected]="p.name === pipeline().render.profileName">{{ p.name }}</option>
                      }
                    </select>
                  </div>
                  <div class="actions">
                    <button type="button" [disabled]="!session().imageId" (click)="render()">
                      Render
                    </button>
                  </div>
                  @if (output().elapsedSeconds !== null) {
                    <p class="hint">Rendered in {{ output().elapsedSeconds | number:'1.1-1' }} s</p>
                  }
                </div>
              </p-accordion-content>
            </p-accordion-panel>

            <!-- ADVANCED panel -->
            <p-accordion-panel value="advanced">
              <p-accordion-header>{{ sectionTitles.advanced }}</p-accordion-header>
              <p-accordion-content>
                <div class="panel-body">
                  <div class="field field-toggle">
                    <label for="pre-upscale">Pre-upscale</label>
                    <input
                      id="pre-upscale"
                      type="checkbox"
                      [checked]="pipeline().advanced.preUpscale"
                      (change)="onPreUpscaleChange($event)"
                    />
                  </div>
                  <div class="field">
                    <label for="upscaler">Upscaler</label>
                    <select
                      id="upscaler"
                      [value]="pipeline().advanced.upscaler"
                      (change)="onUpscalerChange($event)"
                    >
                      @for (opt of upscalerOptions; track opt.value) {
                        <option [value]="opt.value">{{ opt.label }}</option>
                      }
                    </select>
                  </div>
                  <div class="field">
                    <label for="target-mp">Target megapixels</label>
                    <input
                      id="target-mp"
                      type="number"
                      min="0.5"
                      max="20"
                      step="0.5"
                      [value]="pipeline().advanced.targetMP"
                      (change)="onTargetMPChange($event)"
                    />
                  </div>
                  <div class="field">
                    <label>Sharpen <span class="value-badge">{{ pipeline().advanced.sharpen | number:'1.2-2' }}</span></label>
                    <input
                      type="range"
                      min="0"
                      max="2"
                      step="0.01"
                      [value]="pipeline().advanced.sharpen"
                      (change)="onSharpenChange($event)"
                    />
                  </div>
                  <div class="field">
                    <label>Bilateral strength <span class="value-badge">{{ pipeline().advanced.bilateralStrength | number:'1.2-2' }}</span></label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.01"
                      [value]="pipeline().advanced.bilateralStrength"
                      (change)="onBilateralChange($event)"
                    />
                  </div>
                </div>
              </p-accordion-content>
            </p-accordion-panel>

            <!-- OUTPUT panel -->
            <p-accordion-panel value="output">
              <p-accordion-header>{{ sectionTitles.output }}</p-accordion-header>
              <p-accordion-content>
                <div class="panel-body">
                  @if (output().heightmapId) {
                    <p class="hint">Heightmap: <code>{{ output().heightmapId }}</code></p>
                  }
                  <div class="actions actions-stack">
                    <button
                      type="button"
                      [disabled]="!output().heightmapId"
                      (click)="computePlan()"
                    >
                      Compute pass plan
                    </button>
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
              </p-accordion-content>
            </p-accordion-panel>
          </p-accordion>
        </main>

        <!-- Right: persistent preview pane -->
        <aside class="studio-preview">
          <p-card header="Preview" subheader="Session state">
            <div class="preview-tile">
              <h3>Source image</h3>
              @if (session().sourceMeta; as meta) {
                <p>{{ meta.w }} × {{ meta.h }} px · {{ meta.bytes | number }} bytes</p>
                <p class="hash">{{ session().imageHash?.slice(0, 16) }}…</p>
              } @else {
                <p class="muted">Upload an image to begin.</p>
              }
            </div>
            <div class="preview-tile">
              <h3>Heightmap preview</h3>
              @if (output().previewId; as pid) {
                <img [src]="apiClient.blobUrl(pid)" alt="Heightmap preview" style="width:100%;" />
              } @else {
                <p class="muted">Run render to generate a preview.</p>
              }
            </div>
            <div class="preview-tile">
              <h3>Session history</h3>
              @if (session().history.length > 0) {
                <ol class="history-list">
                  @for (entry of session().history.slice(0, 5); track entry.id) {
                    <li>{{ entry.action }}</li>
                  }
                </ol>
              } @else {
                <p class="muted">No actions yet.</p>
              }
            </div>
          </p-card>
        </aside>
      </div>
    </div>
  `,
  styles: `
    :host {
      display: block;
      min-height: calc(100dvh - 5rem);
    }

    .studio-layout {
      display: flex;
      flex-direction: column;
      height: 100%;
      padding: 0.75rem;
      gap: 0.75rem;
    }

    .studio-topbar {
      display: flex;
      align-items: baseline;
      gap: 1rem;
    }

    .eyebrow {
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-size: 0.75rem;
      color: var(--text-muted);
      margin: 0;
    }

    h1 {
      margin: 0;
      font-size: 1.5rem;
    }

    .topbar-meta {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
    }

    .badge {
      padding: 0.2rem 0.6rem;
      border-radius: 999px;
      background: var(--badge-bg);
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--badge-fg);
    }

    .badge.muted {
      color: var(--badge-muted-fg);
      background: var(--badge-muted-bg);
    }

    .badge.success {
      background: var(--badge-success-bg);
      color: var(--badge-success-fg);
    }

    .studio-body {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 22rem;
      gap: 0.75rem;
      flex: 1;
    }

    .studio-controls {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      overflow-y: auto;
    }

    .upload-strip {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      padding: 0.75rem 1rem;
      border: 1px solid var(--border-default);
      border-radius: 0.75rem;
      background: var(--bg-surface);
    }

    .upload-label {
      font-weight: 600;
      font-size: 0.9rem;
      white-space: nowrap;
      color: var(--text-primary);
    }

    .upload-strip input[type="file"] {
      flex: 1;
    }

    .panel-body {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
      padding: 0.75rem 0;
    }

    .field {
      display: flex;
      flex-direction: column;
      gap: 0.3rem;
    }

    .field-toggle {
      flex-direction: row;
      align-items: center;
      gap: 0.5rem;
    }

    .field label {
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-secondary);
      display: flex;
      justify-content: space-between;
    }

    .value-badge {
      font-weight: 400;
      color: var(--text-muted);
    }

    .field select,
    .field input[type="number"] {
      border: 1px solid var(--border-input);
      border-radius: 0.5rem;
      padding: 0.5rem 0.6rem;
      font: inherit;
      background: var(--bg-input);
      color: var(--text-primary);
    }

    .field input[type="range"] {
      width: 100%;
      accent-color: var(--action-bg);
    }

    .field input[type="checkbox"] {
      width: 1.1rem;
      height: 1.1rem;
      accent-color: var(--action-bg);
    }

    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-top: 0.25rem;
    }

    .actions-stack {
      flex-direction: column;
    }

    button {
      border-radius: 0.5rem;
      border: 1px solid var(--action-bg);
      background: var(--action-bg);
      color: var(--action-fg);
      padding: 0.5rem 0.9rem;
      font: inherit;
      cursor: pointer;
      white-space: nowrap;
    }

    button.secondary {
      background: var(--action-secondary-bg);
      color: var(--action-secondary-fg);
      border-color: var(--border-input);
    }

    button:disabled {
      opacity: 0.4;
      cursor: not-allowed;
    }

    .hint {
      margin: 0;
      font-size: 0.8rem;
      color: var(--text-muted);
    }

    .hint code {
      font-family: monospace;
      font-size: 0.75rem;
    }

    .studio-preview {
      overflow-y: auto;
    }

    .preview-tile {
      border: 1px solid var(--border-default);
      border-radius: 0;
      padding: 0.75rem;
      margin-bottom: 0.75rem;
    }

    .preview-tile h3 {
      margin: 0 0 0.4rem;
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-muted);
    }

    .preview-tile p {
      margin: 0;
      font-size: 0.85rem;
    }

    .hash {
      font-family: monospace;
      font-size: 0.75rem;
      color: var(--text-faint);
    }

    .muted {
      color: var(--text-faint);
    }

    .plan-summary {
      margin-top: 0.75rem;
      padding: 0.75rem;
      border: 1px solid var(--border-default);
      border-radius: 0.75rem;
      background: var(--bg-sunken);
    }

    .plan-summary h3 {
      margin: 0 0 0.5rem;
      font-size: 0.9rem;
    }

    .pass-list {
      margin: 0;
      padding: 0 0 0 1rem;
      display: grid;
      gap: 0.3rem;
    }

    .pass-list li {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.85rem;
    }

    .swatch {
      display: inline-block;
      width: 0.75rem;
      height: 0.75rem;
      border-radius: 2px;
      flex-shrink: 0;
    }

    .history-list {
      margin: 0;
      padding: 0 0 0 1rem;
      font-size: 0.8rem;
      color: var(--text-secondary);
    }

    @media (max-width: 1024px) {
      .studio-body {
        grid-template-columns: 1fr;
      }
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
  `,
})
export class StudioShellComponent {
  protected readonly sectionTitles = STUDIO_SECTION_TITLES;
  protected readonly maskBackends = STUDIO_MASK_BACKENDS;
  protected readonly upscalerOptions = STUDIO_UPSCALER_OPTIONS;

  protected readonly sessionTree = inject(SessionTreeService);
  protected readonly sessionService = inject(SessionService);
  protected readonly apiClient = inject(ApiClientService);
  private readonly maskService = inject(MaskService);
  private readonly renderService = inject(RenderService);
  private readonly exportService = inject(ExportService);
  private readonly planService = inject(PlanService);

  protected readonly session = this.sessionTree.session;
  protected readonly pipeline = this.sessionTree.pipeline;
  protected readonly output = this.sessionTree.output;
  protected readonly ui = this.sessionTree.ui;
  protected readonly toasts = computed(() => this.ui().toasts);

  constructor() {
    this.sessionService.loadProfiles();
  }

  protected onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.item(0);
    if (!file) return;
    this.sessionService.uploadImage(file);
    input.value = '';
  }

  protected onMaskBackendChange(event: Event): void {
    const select = event.target as HTMLSelectElement;
    this.maskService.setBackend(select.value as MaskBackend);
  }

  protected onEdgeSoftnessChange(event: Event): void {
    this.maskService.setEdgeSoftness(Number((event.target as HTMLInputElement).value));
  }

  protected createMask(): void {
    this.maskService.createMask();
  }

  protected onDetailBalanceChange(event: Event): void {
    this.renderService.setDetailBalance(Number((event.target as HTMLInputElement).value));
  }

  protected onReliefChange(event: Event): void {
    this.renderService.setRelief(Number((event.target as HTMLInputElement).value));
  }

  protected onMultiresChange(event: Event): void {
    this.renderService.setMultires((event.target as HTMLInputElement).checked);
  }

  protected onProfileChange(event: Event): void {
    const select = event.target as HTMLSelectElement;
    this.sessionService.setProfileName(select.value || null);
  }

  protected render(): void {
    this.renderService.render();
  }

  protected onPreUpscaleChange(event: Event): void {
    const checked = (event.target as HTMLInputElement).checked;
    this.sessionTree.patchState((current) => ({
      ...current,
      pipeline: {
        ...current.pipeline,
        advanced: { ...current.pipeline.advanced, preUpscale: checked },
      },
    }));
  }

  protected onUpscalerChange(event: Event): void {
    const value = (event.target as HTMLSelectElement).value as 'realesrgan' | 'swinir';
    this.sessionTree.patchState((current) => ({
      ...current,
      pipeline: {
        ...current.pipeline,
        advanced: { ...current.pipeline.advanced, upscaler: value },
      },
    }));
  }

  protected onTargetMPChange(event: Event): void {
    const value = Number((event.target as HTMLInputElement).value);
    this.sessionTree.patchState((current) => ({
      ...current,
      pipeline: {
        ...current.pipeline,
        advanced: { ...current.pipeline.advanced, targetMP: value },
      },
    }));
  }

  protected onSharpenChange(event: Event): void {
    this.sessionTree.patchState((current) => ({
      ...current,
      pipeline: {
        ...current.pipeline,
        advanced: {
          ...current.pipeline.advanced,
          sharpen: Number((event.target as HTMLInputElement).value),
        },
      },
    }));
  }

  protected onBilateralChange(event: Event): void {
    this.sessionTree.patchState((current) => ({
      ...current,
      pipeline: {
        ...current.pipeline,
        advanced: {
          ...current.pipeline.advanced,
          bilateralStrength: Number((event.target as HTMLInputElement).value),
        },
      },
    }));
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
}