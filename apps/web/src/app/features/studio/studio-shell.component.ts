import { CommonModule } from '@angular/common';
import { Component, computed, inject } from '@angular/core';

import { Accordion, AccordionContent, AccordionHeader, AccordionPanel } from 'primeng/accordion';
import { Card } from 'primeng/card';

import { ApiClientService } from '../../core/api/api-client.service';
import type { HeightmapSettings } from '../../core/api/api-types';
import { ExportService } from '../../core/state/export.service';
import { MaskService } from '../../core/state/mask.service';
import { PlanService } from '../../core/state/plan.service';
import { RenderService } from '../../core/state/render.service';
import { SculptokService } from '../../core/state/sculptok.service';
import { TargetService } from '../../core/state/target.service';
import { SessionService } from '../../core/state/session.service';
import { SessionTreeService } from '../../core/state/session-tree.service';
import { MaskBackend } from '../../core/state/studio-state';

export const STUDIO_SECTION_TITLES = {
  mask: 'Mask',
  input: 'Pre-sculptok input prep',
  background: 'Background pattern',
  render: 'Render',
  heightmap: 'Heightmap',
  refinement: 'Refinement passes',
  output: 'Output',
} as const;

export const BACKGROUND_PATTERNS: { label: string; value: 'none' | 'guilloche' | 'stripes' | 'dots' | 'halftone' | 'checkers' }[] = [
  { label: '— none —', value: 'none' },
  { label: 'Guilloché', value: 'guilloche' },
  { label: 'Stripes', value: 'stripes' },
  { label: 'Dots', value: 'dots' },
  { label: 'Halftone', value: 'halftone' },
  { label: 'Checkers', value: 'checkers' },
];

export const SIGNATURE_CORNERS: { label: string; value: 'tl' | 'tr' | 'bl' | 'br' }[] = [
  { label: 'Top left', value: 'tl' },
  { label: 'Top right', value: 'tr' },
  { label: 'Bottom left', value: 'bl' },
  { label: 'Bottom right', value: 'br' },
];

export const HEIGHTMAP_POLARITIES: { label: string; value: 'bright_raised' | 'dark_raised' | 'auto' }[] = [
  { label: 'Bright raised (sculptok / meshy)', value: 'bright_raised' },
  { label: 'Dark raised', value: 'dark_raised' },
  { label: 'Auto-detect from corners', value: 'auto' },
];

export const STUDIO_MASK_BACKENDS: { label: string; value: MaskBackend }[] = [
  { label: 'BiRefNet (best quality)', value: 'birefnet' },
  { label: 'RemBG (fast)', value: 'rembg' },
  { label: 'Threshold (no install needed)', value: 'threshold' },
];

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

          <p-accordion [multiple]="true" [value]="['mask', 'render', 'refinement']">
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

            <!-- INPUT PREP panel -->
            <p-accordion-panel value="input">
              <p-accordion-header>{{ sectionTitles.input }}</p-accordion-header>
              <p-accordion-content>
                <div class="panel-body">
                  <p class="hint">
                    Cleans the photo before it reaches sculptok. Default off — turn on
                    when the source is dim, blurry, or has heavy specular highlights.
                  </p>
                  <div class="field field-toggle">
                    <label for="prep-wb">White balance</label>
                    <input id="prep-wb" type="checkbox"
                      [checked]="pipeline().settings.input_white_balance"
                      (change)="onSettingToggle('input_white_balance', $event)" />
                  </div>
                  <div class="field field-toggle">
                    <label for="prep-clahe">CLAHE contrast</label>
                    <input id="prep-clahe" type="checkbox"
                      [checked]="pipeline().settings.input_clahe"
                      (change)="onSettingToggle('input_clahe', $event)" />
                  </div>
                  <div class="field field-toggle">
                    <label for="prep-denoise">Denoise</label>
                    <input id="prep-denoise" type="checkbox"
                      [checked]="pipeline().settings.input_denoise"
                      (change)="onSettingToggle('input_denoise', $event)" />
                  </div>
                  <div class="field field-toggle">
                    <label for="prep-specular">Remove specular highlights</label>
                    <input id="prep-specular" type="checkbox"
                      [checked]="pipeline().settings.input_remove_specular"
                      (change)="onSettingToggle('input_remove_specular', $event)" />
                  </div>
                  <div class="field">
                    <label for="prep-max-dim">Max input dimension (0 = unlimited)</label>
                    <input id="prep-max-dim" type="number" min="0" max="8192" step="64"
                      [value]="pipeline().settings.input_max_dim"
                      (change)="onSettingNumber('input_max_dim', $event)" />
                  </div>
                  <div class="field field-toggle">
                    <label for="prep-auto-orient">Auto-orient face (eye-line level)</label>
                    <input id="prep-auto-orient" type="checkbox"
                      [checked]="pipeline().settings.input_auto_orient_face"
                      (change)="onSettingToggle('input_auto_orient_face', $event)" />
                  </div>
                  <div class="field field-toggle">
                    <label for="prep-auto-crop">Auto-crop to target aspect</label>
                    <input id="prep-auto-crop" type="checkbox"
                      [checked]="pipeline().settings.input_auto_crop"
                      (change)="onSettingToggle('input_auto_crop', $event)" />
                  </div>
                  @if (pipeline().settings.input_auto_crop) {
                    <div class="field field-indented">
                      <label for="prep-auto-crop-aspect">Crop aspect (W/H, 0 = use target)</label>
                      <input id="prep-auto-crop-aspect" type="number" min="0" max="10" step="0.05"
                        [value]="pipeline().settings.input_auto_crop_aspect"
                        (change)="onSettingNumber('input_auto_crop_aspect', $event)" />
                    </div>
                  }
                </div>
              </p-accordion-content>
            </p-accordion-panel>

            <!-- BACKGROUND PATTERN panel -->
            <p-accordion-panel value="background">
              <p-accordion-header>{{ sectionTitles.background }}</p-accordion-header>
              <p-accordion-content>
                <div class="panel-body">
                  <p class="hint">
                    Composites a procedural pattern over the photo's background pixels
                    BEFORE sculptok sees it. Requires the subject mask to be enabled
                    (in the Mask panel) so we know which pixels are background.
                  </p>
                  <div class="field">
                    <label for="bg-pattern">Pattern</label>
                    <select id="bg-pattern"
                      [value]="pipeline().settings.background_pattern"
                      (change)="onSettingValue('background_pattern', $event)">
                      @for (opt of backgroundPatterns; track opt.value) {
                        <option [value]="opt.value">{{ opt.label }}</option>
                      }
                    </select>
                  </div>
                  @if (pipeline().settings.background_pattern !== 'none') {
                    <div class="field">
                      <label>Scale <span class="value-badge">{{ pipeline().settings.background_scale | number:'1.2-2' }}</span></label>
                      <input type="range" min="0.25" max="4" step="0.05"
                        [value]="pipeline().settings.background_scale"
                        (change)="onSettingNumber('background_scale', $event)" />
                    </div>
                    <div class="field">
                      <label>Angle <span class="value-badge">{{ pipeline().settings.background_angle | number:'1.0-0' }}°</span></label>
                      <input type="range" min="-90" max="90" step="1"
                        [value]="pipeline().settings.background_angle"
                        (change)="onSettingNumber('background_angle', $event)" />
                    </div>
                    <div class="field">
                      <label>Intensity <span class="value-badge">{{ pipeline().settings.background_intensity | number:'1.2-2' }}</span></label>
                      <input type="range" min="0" max="1" step="0.01"
                        [value]="pipeline().settings.background_intensity"
                        (change)="onSettingNumber('background_intensity', $event)" />
                    </div>
                    <div class="field">
                      <label for="bg-seed">Seed</label>
                      <input id="bg-seed" type="number" min="0" max="2147483647" step="1"
                        [value]="pipeline().settings.background_seed"
                        (change)="onSettingNumber('background_seed', $event)" />
                    </div>
                  }
                </div>
              </p-accordion-content>
            </p-accordion-panel>

            <!-- RENDER panel -->
            <p-accordion-panel value="render">
              <p-accordion-header>{{ sectionTitles.render }}</p-accordion-header>
              <p-accordion-content>
                <div class="panel-body">
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
                    <button type="button"
                      [disabled]="!canRender()"
                      (click)="render()">
                      Render
                    </button>
                  </div>
                  @if (!pipeline().settings.external_heightmap_path && session().imageId) {
                    <p class="hint">
                      Render needs a heightmap source. Open the <strong>Heightmap</strong>
                      panel and either click <strong>Generate via Sculptok</strong> or
                      drop a heightmap PNG path into the source field.
                    </p>
                  }
                  <div class="actions">
                    <button type="button" class="secondary" (click)="onSaveProfile()">
                      Save current settings as profile…
                    </button>
                    @if (pipeline().render.profileName) {
                      <button type="button" class="secondary" (click)="onDeleteProfile()">
                        Delete this profile
                      </button>
                    }
                  </div>
                  @if (output().elapsedSeconds !== null) {
                    <p class="hint">Rendered in {{ output().elapsedSeconds | number:'1.1-1' }} s</p>
                  }
                </div>
              </p-accordion-content>
            </p-accordion-panel>

            <!-- HEIGHTMAP panel -->
            <p-accordion-panel value="heightmap">
              <p-accordion-header>{{ sectionTitles.heightmap }}</p-accordion-header>
              <p-accordion-content>
                <div class="panel-body">
                  <!-- Target preset -->
                  <div class="field">
                    <label for="hm-target">Target object</label>
                    <select id="hm-target"
                      [value]="targetService.active() ?? ''"
                      (change)="onTargetChange($event)">
                      <option value="">— pick a target —</option>
                      @for (t of targetService.presets(); track t.name) {
                        <option [value]="t.name" [selected]="t.name === targetService.active()">
                          {{ t.display_name }} ({{ t.print_width_mm }}×{{ t.print_height_mm }} mm{{ t.polarity_invert ? ', invert' : '' }})
                        </option>
                      }
                    </select>
                  </div>

                  <!-- Sculptok auto-pull -->
                  <div class="field">
                    <label>Heightmap source</label>
                    @if (pipeline().settings.external_heightmap_path) {
                      <p class="hint">
                        Loaded: <code>{{ pipeline().settings.external_heightmap_path }}</code>
                      </p>
                    } @else {
                      <p class="hint">No heightmap loaded yet — generate via Sculptok or render and the response will populate this field.</p>
                    }
                  </div>
                  <div class="field">
                    <div class="actions">
                      <button type="button"
                        [disabled]="!session().imageId || !sculptokService.credits()?.configured || sculptokService.inFlight()"
                        (click)="sculptokGenerate()">
                        @if (sculptokService.inFlight()) { Sculptok generating… } @else { Generate via Sculptok }
                      </button>
                    </div>
                    <div class="field">
                      <label for="hm-upload">Or upload a heightmap PNG</label>
                      <input id="hm-upload" type="file"
                        accept="image/png,image/tiff"
                        (change)="onHeightmapFileSelected($event)" />
                    </div>
                    @if (sculptokService.credits(); as c) {
                      @if (c.configured) {
                        <p class="hint">
                          Sculptok credits: {{ c.balance }} (each pro/2k call costs {{ c.cost_pro_2k }})
                        </p>
                      } @else {
                        <p class="hint">
                          Sculptok API key not configured on the server. Set <code>SCULPTOK_API_KEY</code> or
                          add <code>credentials.sculptok_api_key</code> to <code>~/.mopa-heightmap/settings.json</code>.
                        </p>
                      }
                    }
                  </div>

                  <div class="field">
                    <label for="hm-polarity">Source polarity</label>
                    <select id="hm-polarity"
                      [value]="pipeline().settings.external_heightmap_polarity"
                      (change)="onSettingValue('external_heightmap_polarity', $event)">
                      @for (opt of heightmapPolarities; track opt.value) {
                        <option [value]="opt.value">{{ opt.label }}</option>
                      }
                    </select>
                  </div>
                  <div class="field field-toggle">
                    <label for="hm-invert">Invert (signet ring / recessed)</label>
                    <input id="hm-invert" type="checkbox"
                      [checked]="pipeline().settings.polarity_invert"
                      (change)="onSettingToggle('polarity_invert', $event)" />
                  </div>
                  <div class="field field-toggle">
                    <label for="hm-bid">Black is deep</label>
                    <input id="hm-bid" type="checkbox"
                      [checked]="pipeline().settings.black_is_deep"
                      (change)="onSettingToggle('black_is_deep', $event)" />
                  </div>
                  <div class="field">
                    <label>Background value <span class="value-badge">{{ pipeline().settings.background_value | number:'1.2-2' }}</span></label>
                    <input type="range" min="0" max="1" step="0.01"
                      [value]="pipeline().settings.background_value"
                      (change)="onSettingNumber('background_value', $event)" />
                  </div>
                </div>
              </p-accordion-content>
            </p-accordion-panel>

            <!-- REFINEMENT panel -->
            <p-accordion-panel value="refinement">
              <p-accordion-header>{{ sectionTitles.refinement }}</p-accordion-header>
              <p-accordion-content>
                <div class="panel-body">
                  <p class="hint">
                    Refinement passes add separate physical features on top of the carved
                    relief — they don't subdivide the depth budget.
                  </p>

                  <div class="field field-toggle">
                    <label for="ref-mask">Subject mask deliverable</label>
                    <input id="ref-mask" type="checkbox"
                      [checked]="pipeline().settings.subject_mask_enabled"
                      (change)="onSettingToggle('subject_mask_enabled', $event)" />
                  </div>

                  <div class="field field-toggle">
                    <label for="ref-preclean">Pre-clean pass</label>
                    <input id="ref-preclean" type="checkbox"
                      [checked]="pipeline().settings.pre_clean_enabled"
                      (change)="onSettingToggle('pre_clean_enabled', $event)" />
                  </div>

                  <div class="field field-toggle">
                    <label for="ref-tonal">Photo-tonal overlay</label>
                    <input id="ref-tonal" type="checkbox"
                      [checked]="pipeline().settings.photo_tonal_enabled"
                      (change)="onSettingToggle('photo_tonal_enabled', $event)" />
                  </div>
                  @if (pipeline().settings.photo_tonal_enabled) {
                    <div class="field field-indented">
                      <label>Strength <span class="value-badge">{{ pipeline().settings.photo_tonal_strength | number:'1.2-2' }}</span></label>
                      <input type="range" min="0" max="1" step="0.01"
                        [value]="pipeline().settings.photo_tonal_strength"
                        (change)="onSettingNumber('photo_tonal_strength', $event)" />
                    </div>
                    <div class="field field-indented field-toggle">
                      <label for="ref-tonal-invert">Invert (light = engrave)</label>
                      <input id="ref-tonal-invert" type="checkbox"
                        [checked]="pipeline().settings.photo_tonal_invert"
                        (change)="onSettingToggle('photo_tonal_invert', $event)" />
                    </div>
                  }

                  <div class="field">
                    <label for="ref-sig-text">Signature text</label>
                    <input id="ref-sig-text" type="text" maxlength="64"
                      placeholder="e.g. JB 2026"
                      [value]="pipeline().settings.signature_text"
                      (change)="onSettingValue('signature_text', $event)" />
                  </div>
                  @if (pipeline().settings.signature_text) {
                    <div class="field field-indented">
                      <label for="ref-sig-corner">Corner</label>
                      <select id="ref-sig-corner"
                        [value]="pipeline().settings.signature_corner"
                        (change)="onSettingValue('signature_corner', $event)">
                        @for (opt of signatureCorners; track opt.value) {
                          <option [value]="opt.value">{{ opt.label }}</option>
                        }
                      </select>
                    </div>
                  }

                  <div class="field field-toggle">
                    <label for="ref-dither">Output dither (8-bit collapse)</label>
                    <input id="ref-dither" type="checkbox"
                      [checked]="pipeline().settings.dither"
                      (change)="onSettingToggle('dither', $event)" />
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
  protected readonly heightmapPolarities = HEIGHTMAP_POLARITIES;
  protected readonly signatureCorners = SIGNATURE_CORNERS;
  protected readonly backgroundPatterns = BACKGROUND_PATTERNS;

  protected readonly sessionTree = inject(SessionTreeService);
  protected readonly sessionService = inject(SessionService);
  protected readonly apiClient = inject(ApiClientService);
  protected readonly sculptokService = inject(SculptokService);
  protected readonly targetService = inject(TargetService);
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
    this.sculptokService.loadCredits();
    this.targetService.loadPresets();
  }

  protected sculptokGenerate(): void {
    this.sculptokService.generate();
  }

  protected onHeightmapFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.item(0);
    if (!file) return;
    this.sculptokService.uploadHeightmap(file);
    input.value = '';
  }

  protected onTargetChange(event: Event): void {
    const value = (event.target as HTMLSelectElement).value;
    if (value) this.targetService.apply(value);
  }

  protected onSaveProfile(): void {
    const suggested = this.pipeline().render.profileName ?? 'my-profile';
    const name = window.prompt('Save current settings as profile (name):', suggested);
    if (!name) return;
    const overwrite = this.sessionService.profiles().some((p) => p.name === name);
    if (overwrite && !window.confirm(`Profile "${name}" exists. Overwrite?`)) return;
    this.sessionService.saveCurrentAsProfile(name.trim(), { overwrite });
  }

  protected onDeleteProfile(): void {
    const name = this.pipeline().render.profileName;
    if (!name) return;
    if (!window.confirm(`Delete profile "${name}"? This only removes user-scope profiles; shipped profiles are protected.`)) return;
    this.sessionService.deleteCurrentProfile();
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

  protected onProfileChange(event: Event): void {
    const select = event.target as HTMLSelectElement;
    this.sessionService.setProfileName(select.value || null);
  }

  protected render(): void {
    this.renderService.render();
  }

  /**
   * Render is allowed when an image is uploaded AND a heightmap source
   * is configured (sculptok-generated or user-supplied).
   */
  protected canRender(): boolean {
    const state = this.sessionTree.state();
    return Boolean(state.session.imageId && state.pipeline.settings.external_heightmap_path);
  }

  /** Generic toggle handler for boolean ``HeightmapSettings`` keys. */
  protected onSettingToggle<K extends keyof HeightmapSettings>(key: K, event: Event): void {
    const checked = (event.target as HTMLInputElement).checked;
    this.renderService.patchSettings(key, checked as HeightmapSettings[K]);
  }

  /** Generic numeric handler (sliders, number inputs). */
  protected onSettingNumber<K extends keyof HeightmapSettings>(key: K, event: Event): void {
    const value = Number((event.target as HTMLInputElement).value);
    this.renderService.patchSettings(key, value as HeightmapSettings[K]);
  }

  /** Generic string-or-enum handler (selects, text inputs). */
  protected onSettingValue<K extends keyof HeightmapSettings>(key: K, event: Event): void {
    const value = (event.target as HTMLInputElement | HTMLSelectElement).value;
    this.renderService.patchSettings(key, value as HeightmapSettings[K]);
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