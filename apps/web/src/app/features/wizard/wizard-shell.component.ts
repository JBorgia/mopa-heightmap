import { CommonModule, isPlatformBrowser } from '@angular/common';
import {
  Component,
  DestroyRef,
  PLATFORM_ID,
  computed,
  effect,
  inject,
  signal,
} from '@angular/core';

import { SharedModule } from 'primeng/api';
import { Card } from 'primeng/card';
import { Splitter } from 'primeng/splitter';

import { ApiClientService } from '../../core/api/api-client.service';
import type { HeightmapSettings } from '../../core/api/api-types';
import { InfoTipComponent } from '../../core/ui/info-tip.component';
import { ExportService } from '../../core/state/export.service';
import { MaskService } from '../../core/state/mask.service';
import { PlanService } from '../../core/state/plan.service';
import { RenderService } from '../../core/state/render.service';
import { SculptokService } from '../../core/state/sculptok.service';
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

/**
 * Per-step optionality flag. The mask step is a separate deliverable
 * (LightBurn handles subject isolation at engrave time per the project
 * memory) so it must NOT block forward progress. Everything else is
 * required to produce an export.
 */
export const WIZARD_PAGE_OPTIONAL = [false, true, false, false, false] as const;

export const WIZARD_STAGE_SUMMARIES = [
  'Upload a photo of the subject you want to engrave.',
  'Optional — produce a subject mask deliverable. LightBurn applies it at engrave time, so you can skip this step entirely if you only need the heightmap.',
  'Load a heightmap source (sculptok or your own PNG), optionally clean the photo before sculptok sees it, and pick refinement layers to ship in the bundle.',
  'Choose a material profile. The pass plan computes automatically when both the heightmap and a profile are ready.',
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
  imports: [CommonModule, SharedModule, Card, Splitter, InfoTipComponent],
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
                  [class.complete]="pageStatus(index) === 'complete'"
                  [class.optional]="wizardPageOptional[index]"
                  [attr.aria-label]="label + ' — ' + pageStatusLabel(index)"
                  [attr.title]="pageStatusLabel(index)"
                  (click)="selectPage(index)"
                >
                  <span class="page-chip-status" aria-hidden="true">{{ pageStatusIcon(index) }}</span>
                  {{ label }}
                  @if (wizardPageOptional[index]) {
                    <span class="page-chip-tag">optional</span>
                  }
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

              <!-- Page 1: Subject / Mask (optional — separate LightBurn deliverable) -->
              @if (ui().wizardPage === 1) {
                <h2>{{ wizardPageLabels[1] }} <span class="step-tag">optional</span></h2>
                <p>{{ stageSummaries[1] }}</p>
                @if (!session().imageId) {
                  <p class="muted">Upload an image first (step 1) to enable the mask backend.</p>
                }
                <div class="wizard-controls">
                  <div class="control-group">
                    <label for="wiz-mask-backend">
                      Mask backend
                      <app-info-tip label="Mask backend"
                        text="Which algorithm separates the subject from the background. BiRefNet gives the cleanest edges but needs PyTorch + CUDA. Rembg is fast and CPU-friendly. Threshold uses simple luminance and works without any ML deps — fine for high-contrast photos."></app-info-tip>
                    </label>
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
                      <app-info-tip label="Edge softness"
                        text="How much to feather the mask boundary (0 = hard cut, 1 = heavy blur). Soft edges blend the subject into a procedural background; hard edges keep the silhouette crisp for vector cuts."></app-info-tip>
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
                        Refine the boundary by adding flood-fill markers. Each shortcut
                        adds an include (positive) or exclude (negative) seed at a fixed
                        normalised position on the source image:
                      </p>
                      <div class="control-actions">
                        <button
                          type="button"
                          class="secondary"
                          (click)="clickRefineAt(0.5, 0.5, 'positive')"
                        >
                          Include centre
                        </button>
                        <button
                          type="button"
                          class="secondary"
                          (click)="clickRefineAt(0.05, 0.05, 'negative')"
                        >
                          Exclude top-left
                        </button>
                        <button
                          type="button"
                          class="secondary"
                          (click)="clickRefineAt(0.95, 0.95, 'negative')"
                        >
                          Exclude bottom-right
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
                @if (!session().imageId) {
                  <p class="muted">Upload an image first (step 1) before configuring the heightmap.</p>
                }
                <div class="wizard-controls">

                  <fieldset class="control-section primary">
                    <legend>
                      Heightmap source <span class="step-tag required">required</span>
                      <app-info-tip label="Heightmap source"
                        text="A greyscale PNG where pixel brightness encodes depth — bright = raised, dark = engraved deep. This is what LightBurn's 3D Sliced mode reads to carve the relief. Sculptok generates one from your photo via their API; otherwise upload your own (Meshy, hand-painted, etc.)."></app-info-tip>
                    </legend>
                    <p class="muted small">
                      The depth pass uses this PNG verbatim. Generate via the Sculptok API
                      or upload one you've already produced.
                    </p>
                    @if (pipeline().settings.external_heightmap_path) {
                      <p class="muted">
                        Loaded: <code>{{ pipeline().settings.external_heightmap_path }}</code>
                      </p>
                    }
                    <div class="control-group">
                      <button type="button"
                        [disabled]="!session().imageId || !sculptokService.credits()?.configured || sculptokService.inFlight()"
                        (click)="sculptokGenerate()">
                        @if (sculptokService.inFlight()) { Sculptok generating… } @else { Generate via Sculptok }
                      </button>
                      @if (sculptokService.credits(); as c) {
                        @if (c.configured) {
                          <p class="muted">Sculptok credits: {{ c.balance }}</p>
                        } @else {
                          <p class="muted">
                            Sculptok API key not configured on the server. Upload a PNG below
                            instead, or configure <code>SCULPTOK_API_KEY</code>.
                          </p>
                        }
                      }
                    </div>
                    <div class="control-group">
                      <label for="wiz-heightmap-upload">Or upload a heightmap PNG</label>
                      <input id="wiz-heightmap-upload" type="file"
                        accept="image/png,image/tiff"
                        (change)="onHeightmapFileSelected($event)" />
                    </div>
                  </fieldset>

                  <fieldset class="control-section">
                    <legend>Pre-sculptok prep <span class="step-tag">optional</span></legend>
                    <p class="muted small">
                      Cleans the photo before sculptok sees it. Defaults are off — turn on
                      when the source is dim, blurry, or has heavy specular highlights.
                    </p>
                    <label class="control-toggle">
                      <input type="checkbox"
                        [checked]="pipeline().settings.input_clahe"
                        (change)="onSettingToggle('input_clahe', $event)" />
                      <span>
                        CLAHE contrast
                        <app-info-tip label="CLAHE contrast"
                          text="Contrast Limited Adaptive Histogram Equalisation. Boosts local contrast in dim or muddy photos so sculptok sees more facial detail. Don't enable on already-bright studio shots — it'll posterise."></app-info-tip>
                      </span>
                    </label>
                    <label class="control-toggle">
                      <input type="checkbox"
                        [checked]="pipeline().settings.input_denoise"
                        (change)="onSettingToggle('input_denoise', $event)" />
                      <span>
                        Denoise
                        <app-info-tip label="Denoise"
                          text="Bilateral denoise. Smooths out sensor grain and JPEG noise without losing edges. Useful for high-ISO phone shots; skip for clean stock photography."></app-info-tip>
                      </span>
                    </label>
                    <label class="control-toggle">
                      <input type="checkbox"
                        [checked]="pipeline().settings.input_remove_specular"
                        (change)="onSettingToggle('input_remove_specular', $event)" />
                      <span>
                        Remove specular highlights
                        <app-info-tip label="Remove specular highlights"
                          text="Caps the brightest pixels (typically reflections off skin or jewellery) so sculptok doesn't read them as raised features. Helps with shiny foreheads, wet lips, polished metal."></app-info-tip>
                      </span>
                    </label>

                    <!-- Background replace: uses the subject mask to scrub
                         the photo's background before sculptok sees it.
                         Same mask is also shipped as a deliverable when
                         "Subject mask deliverable" is on. -->
                    <div class="control-group">
                      <label for="wiz-bg-pattern">
                        Replace background before sculptok
                        <app-info-tip label="Replace background"
                          text="Uses the subject mask (computed automatically) to scrub the photo's background to a flat colour or pattern before sculptok generates the heightmap. With a busy background removed, sculptok focuses on the subject and produces cleaner depth. The same mask is shipped as the LightBurn deliverable when 'Subject mask deliverable' is on, so you don't pay for it twice."></app-info-tip>
                      </label>
                      <select id="wiz-bg-pattern"
                        [value]="pipeline().settings.background_pattern"
                        (change)="onBackgroundPatternChange($event)">
                        <option value="none">— don't replace —</option>
                        <option value="solid_black">Solid black</option>
                        <option value="solid_white">Solid white</option>
                        <option value="solid_grey">Solid mid-grey</option>
                        <option value="guilloche">Guilloché (decorative)</option>
                        <option value="stripes">Stripes (decorative)</option>
                        <option value="dots">Dots (decorative)</option>
                        <option value="halftone">Halftone (decorative)</option>
                        <option value="checkers">Checkers (decorative)</option>
                      </select>
                      @if (pipeline().settings.background_pattern !== 'none' && !pipeline().settings.subject_mask_enabled) {
                        <p class="muted small disabled-hint">
                          Background replace needs a subject mask — enabling it
                          automatically when you click <strong>Generate via Sculptok</strong>.
                        </p>
                      }
                    </div>
                  </fieldset>

                  <fieldset class="control-section">
                    <legend>Refinement passes <span class="step-tag">optional</span></legend>
                    <p class="muted small">
                      Extra layers to ship in the .lbrn2 bundle. Each adds a separate
                      physical pass — they don't subdivide the depth budget.
                    </p>
                    <label class="control-toggle">
                      <input type="checkbox"
                        [checked]="pipeline().settings.subject_mask_enabled"
                        (change)="onSettingToggle('subject_mask_enabled', $event)" />
                      <span>
                        Subject mask deliverable
                        <app-info-tip label="Subject mask deliverable"
                          text="Computes a silhouette mask during render and ships it alongside the heightmap. LightBurn uses it to limit the engraving to the subject area only. Required if you've selected a procedural background pattern."></app-info-tip>
                      </span>
                    </label>
                    <label class="control-toggle">
                      <input type="checkbox"
                        [checked]="pipeline().settings.pre_clean_enabled"
                        (change)="onSettingToggle('pre_clean_enabled', $event)" />
                      <span>
                        Pre-clean pass
                        <app-info-tip label="Pre-clean pass"
                          text="Defocused full-frame raster pass run BEFORE the depth carve. Burns off oxide, oils, and surface contamination so the relief lands on bare metal. Adds engrave time but improves tonal consistency."></app-info-tip>
                      </span>
                    </label>
                    <label class="control-toggle">
                      <input type="checkbox"
                        [checked]="pipeline().settings.photo_tonal_enabled"
                        (change)="onSettingToggle('photo_tonal_enabled', $event)" />
                      <span>
                        Photo-tonal overlay
                        <app-info-tip label="Photo-tonal overlay"
                          text="Low-power dithered pass that fires the photo's luminance back over the carved relief. Adds skin tone, hair shading, and clothing patterns that pure depth misses. Tonal-only — does NOT carve depth."></app-info-tip>
                      </span>
                    </label>
                    <label class="control-toggle">
                      <input type="checkbox"
                        [checked]="pipeline().settings.polarity_invert"
                        (change)="onSettingToggle('polarity_invert', $event)" />
                      <span>
                        Polarity invert (signet ring)
                        <app-info-tip label="Polarity invert"
                          text="Flips the heightmap so the subject engraves DEEP and the background stays at the surface. Use for signet rings, intaglio seals, or any inverted relief where the design is recessed."></app-info-tip>
                      </span>
                    </label>
                    <div class="control-group">
                      <label for="wiz-sig-text">
                        Signature text
                        <app-info-tip label="Signature text"
                          text="Optional vector text engraved in a corner of the piece. Leave blank to omit. Useful for an artist mark, date, or serial number."></app-info-tip>
                      </label>
                      <input id="wiz-sig-text" type="text" maxlength="64"
                        placeholder="e.g. JB 2026"
                        [value]="pipeline().settings.signature_text"
                        (change)="onSettingValue('signature_text', $event)" />
                    </div>
                  </fieldset>

                  <div class="control-actions">
                    <button type="button"
                      [disabled]="!canRender() || renderService.inFlight()"
                      (click)="renderPreview()">
                      @if (renderService.inFlight()) {
                        Rendering…
                      } @else if (output().heightmapId) {
                        Re-render preview
                      } @else {
                        Render preview
                      }
                    </button>
                  </div>
                  @if (!session().imageId) {
                    <p class="muted disabled-hint">Upload an image (step 1) before rendering.</p>
                  } @else if (!pipeline().settings.external_heightmap_path) {
                    <p class="muted disabled-hint">Pick a heightmap source above to enable render.</p>
                  } @else if (renderService.inFlight()) {
                    <p class="muted disabled-hint">This may take 5–15 seconds depending on image size.</p>
                  }
                  @if (output().elapsedSeconds !== null && !renderService.inFlight()) {
                    <p class="muted">Rendered in {{ output().elapsedSeconds | number: '1.2-2' }} s — ready for step 4.</p>
                  }
                </div>
              }

              <!-- Page 3: Material & Passes -->
              @if (ui().wizardPage === 3) {
                <h2>{{ wizardPageLabels[3] }}</h2>
                <p>{{ stageSummaries[3] }}</p>
                <div class="wizard-controls">
                  <div class="control-group">
                    <label for="wiz-profile">
                      Material profile
                      <app-info-tip label="Material profile"
                        text="A bundle of laser parameters (speed, power, frequency, line interval, pass count) tuned for a specific material on a specific machine. Determines depth-per-pass and the overall pass plan. Pick the one that matches your stock — wrong profile = wrong depth."></app-info-tip>
                    </label>
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
                      [disabled]="!output().heightmapId || planService.inFlight()"
                      (click)="computePlan()"
                    >
                      @if (planService.inFlight()) {
                        Computing…
                      } @else if (output().plan) {
                        Recompute pass plan
                      } @else {
                        Compute pass plan
                      }
                    </button>
                  </div>
                  @if (!output().heightmapId) {
                    <p class="muted disabled-hint">
                      Render the heightmap on step 3 first — the pass plan needs it.
                    </p>
                  } @else if (!pipeline().render.profileName) {
                    <p class="muted disabled-hint">
                      Pick a material profile above; the plan computes automatically.
                    </p>
                  } @else if (!output().plan && !planService.inFlight()) {
                    @if (autoPlanAttempted()) {
                      <p class="muted disabled-hint">
                        Auto-compute didn't produce a plan — click above to retry.
                      </p>
                    } @else {
                      <p class="muted disabled-hint">
                        The plan will compute automatically — this may take a moment.
                      </p>
                    }
                  }
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
                      <dd>
                        @if (session().imageId) {
                          Uploaded ✓
                        } @else {
                          <button type="button" class="link-button" (click)="selectPage(0)">Not uploaded — go to step 1 →</button>
                        }
                      </dd>
                    </div>
                    <div>
                      <dt>Mask <span class="step-tag">optional</span></dt>
                      <dd>{{ pipeline().mask.maskId ? 'Ready — ' + (pipeline().mask.coveragePct | number: '1.1-1') + '% coverage' : 'Skipped (LightBurn handles isolation)' }}</dd>
                    </div>
                    <div>
                      <dt>Heightmap</dt>
                      <dd>
                        @if (output().heightmapId) {
                          Rendered ✓
                        } @else {
                          <button type="button" class="link-button" (click)="selectPage(2)">Not rendered — go to step 3 →</button>
                        }
                      </dd>
                    </div>
                    <div>
                      <dt>Pass plan</dt>
                      <dd>
                        @if (output().plan) {
                          {{ output().plan!.passes.length }} passes ready
                        } @else {
                          <button type="button" class="link-button" (click)="selectPage(3)">Not computed — go to step 4 →</button>
                        }
                      </dd>
                    </div>
                  </dl>
                  @if (!output().heightmapId) {
                    <p class="muted disabled-hint export-banner">
                      Render the heightmap on step 3 to enable PNG and STL — the pass plan
                      then auto-computes and unlocks .lbrn2.
                    </p>
                  } @else if (!output().plan) {
                    <p class="muted disabled-hint export-banner">
                      Pass plan still computing — .lbrn2 will be available once it's ready.
                    </p>
                  }
                  <fieldset class="export-picker">
                    <legend>Choose what to bundle</legend>
                    <label class="control-toggle"
                           [class.disabled]="!output().heightmapId">
                      <input type="checkbox"
                        [disabled]="!output().heightmapId"
                        [checked]="exportSelections().png"
                        (change)="toggleExport('png', $event)" />
                      <span>
                        <strong>PNG</strong> — 16-bit heightmap image
                        <app-info-tip label="PNG export"
                          text="The raw rendered heightmap as a 16-bit greyscale PNG. Drop straight into LightBurn's Image trace or feed into another tool. This is the depth data without the LightBurn project structure around it."></app-info-tip>
                      </span>
                    </label>
                    <label class="control-toggle"
                           [class.disabled]="!output().plan">
                      <input type="checkbox"
                        [disabled]="!output().plan"
                        [checked]="exportSelections().lbrn2"
                        (change)="toggleExport('lbrn2', $event)" />
                      <span>
                        <strong>.lbrn2</strong> — LightBurn project + per-pass PNGs
                        <app-info-tip label="LightBurn .lbrn2"
                          text="A complete LightBurn project file with the heightmap and every refinement pass already laid out as separate layers. Open in LightBurn 1.7+ and the cut settings are pre-filled from the material profile. The download is a zip containing the .lbrn2 file plus the PNGs it references — extract it before opening."></app-info-tip>
                        @if (!output().plan) {
                          <span class="muted small">(needs pass plan)</span>
                        }
                      </span>
                    </label>
                    <label class="control-toggle"
                           [class.disabled]="!output().heightmapId">
                      <input type="checkbox"
                        [disabled]="!output().heightmapId"
                        [checked]="exportSelections().stl"
                        (change)="toggleExport('stl', $event)" />
                      <span>
                        <strong>.stl</strong> — 3D mesh for preview / printing
                        <app-info-tip label="STL export"
                          text="Triangulated 3D mesh. Useful for previewing the relief in a viewer (MeshLab, fusion 360, Blender) before laser time, or for 3D-printing the design directly. Heads-up: a 1920×1280 heightmap produces ~5M triangles, around 245 MB — bundles can be slow."></app-info-tip>
                      </span>
                    </label>
                  </fieldset>
                  <p class="muted small">
                    Submit (below) bundles the checked formats into a single
                    <code>mopa_export.zip</code> you can drop into any directory.
                  </p>
                </div>
              }

              <div class="wizard-actions">
                <button type="button" class="secondary" (click)="previousPage()" [disabled]="ui().wizardPage === 0">
                  ← Previous
                </button>
                @if (ui().wizardPage === wizardPageLabels.length - 1) {
                  <button type="button"
                    [disabled]="!canSubmitBundle() || exportService.bundleInFlight()"
                    (click)="submitBundle()">
                    @if (exportService.bundleInFlight()) {
                      Bundling…
                    } @else {
                      Submit — Download {{ submitButtonSummary() }}
                    }
                  </button>
                } @else {
                  <button type="button" (click)="nextPage()">Next →</button>
                }
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
            @if (session().imageId; as imgId) {
              <img [src]="blobUrl(imgId)" alt="Source photo" style="width:100%; display:block;" />
            }
            @if (session().sourceMeta; as meta) {
              <p>{{ meta.w }} × {{ meta.h }} px</p>
            } @else {
              <p class="muted">No image uploaded yet.</p>
            }
          </div>

          @if (output().sculptokInputId; as sid) {
            <div class="preview-tile">
              <h3>Sculptok input <span class="step-tag">uploaded</span></h3>
              <img [src]="blobUrl(sid)" alt="The photo Sculptok actually saw"
                   style="width:100%; display:block;" />
              <p class="muted small">
                After pre-prep + bg-replace. This is the exact image sculptok
                generated the heightmap from.
              </p>
            </div>
          }
          @if (output().conditionedId; as cid) {
            <div class="preview-tile">
              <h3>Prepped photo</h3>
              <img [src]="blobUrl(cid)" alt="Photo after pre-sculptok prep + composite"
                   style="width:100%; display:block;" />
              <p class="muted small">After CLAHE / denoise / specular / auto-orient / auto-crop / background composite (render-time).</p>
            </div>
          }

          @if (pipeline().mask.maskId; as mid) {
            <div class="preview-tile">
              <h3>Subject mask</h3>
              <img [src]="blobUrl(mid)" alt="Subject mask"
                   style="width:100%; display:block; background:#000;" />
              <p class="muted small">{{ pipeline().mask.coveragePct | number: '1.1-1' }}% coverage.</p>
            </div>
          } @else if (output().renderMaskId; as rmid) {
            <div class="preview-tile">
              <h3>Subject mask <span class="step-tag">render-time</span></h3>
              <img [src]="blobUrl(rmid)" alt="Render-time subject mask"
                   style="width:100%; display:block; background:#000;" />
              <p class="muted small">Computed during render for the background composite.</p>
            </div>
          }

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

    .page-chip {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.4rem;
      flex-wrap: wrap;
    }

    .page-chip-status {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 1.1rem;
      height: 1.1rem;
      border-radius: 999px;
      font-size: 0.7rem;
      font-weight: 700;
      background: var(--bg-sunken);
      color: var(--text-muted);
      border: 1px solid var(--border-input);
      flex-shrink: 0;
    }

    .page-chip.complete:not(.active) {
      border-color: color-mix(in srgb, #27ae60 50%, var(--border-input));
    }

    .page-chip.complete .page-chip-status {
      background: #27ae60;
      color: white;
      border-color: #27ae60;
    }

    .page-chip.active.complete .page-chip-status {
      background: white;
      color: #27ae60;
      border-color: white;
    }

    .page-chip-tag {
      font-size: 0.65rem;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      opacity: 0.7;
    }

    .page-chip.active,
    button:not(.secondary):not(.page-chip):not(.toast-dismiss):not(.link-button) {
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

    .step-tag {
      display: inline-block;
      font-size: 0.65rem;
      font-weight: 600;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      padding: 0.1rem 0.45rem;
      border-radius: 999px;
      background: var(--bg-sunken);
      color: var(--text-muted);
      border: 1px solid var(--border-input);
      vertical-align: middle;
      margin-left: 0.4rem;
    }

    .step-tag.required {
      background: color-mix(in srgb, var(--action-bg) 12%, var(--bg-surface));
      color: var(--action-bg);
      border-color: color-mix(in srgb, var(--action-bg) 50%, var(--border-input));
    }

    .control-section {
      display: grid;
      gap: 0.55rem;
      padding: 0.85rem 1rem 1rem;
      border: 1px solid var(--border-default);
      border-radius: 0.75rem;
      background: var(--bg-surface);
      margin: 0;
      /* Grid children default to min-content sizing; without min-width:0 a
         long server path (in <code>) forces the fieldset wider than its
         parent and the whole wizard pane horizontally scrolls. */
      min-width: 0;
    }

    .control-section code {
      overflow-wrap: anywhere;
      word-break: break-all;
      font-size: 0.75rem;
      display: inline-block;
      max-width: 100%;
    }

    .control-section p {
      margin: 0;
      overflow-wrap: anywhere;
    }

    .control-section input[type="file"] {
      max-width: 100%;
    }

    .control-section legend {
      padding: 0 0.4rem;
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-primary);
    }

    .control-section.primary {
      border-color: color-mix(in srgb, var(--action-bg) 50%, var(--border-default));
      background: color-mix(in srgb, var(--action-bg) 4%, var(--bg-surface));
    }

    .control-toggle {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-size: 0.9rem;
      cursor: pointer;
    }

    .control-toggle input[type="checkbox"] {
      width: 1.05rem;
      height: 1.05rem;
      accent-color: var(--action-bg);
    }

    .disabled-hint {
      font-size: 0.85rem;
      color: var(--text-muted);
      margin: 0.25rem 0 0;
    }

    .muted.small {
      font-size: 0.75rem;
    }

    .export-row {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
    }

    .export-row button {
      flex: 0 0 auto;
    }

    .export-banner {
      padding: 0.6rem 0.85rem;
      border: 1px dashed color-mix(in srgb, var(--text-muted) 40%, var(--border-default));
      border-radius: 0.5rem;
      background: color-mix(in srgb, var(--text-muted) 5%, var(--bg-surface));
      margin: 0;
    }

    .export-picker {
      display: grid;
      gap: 0.5rem;
      padding: 0.85rem 1rem;
      border: 1px solid var(--border-default);
      border-radius: 0.75rem;
      background: var(--bg-surface);
      margin: 0;
    }

    .export-picker legend {
      padding: 0 0.4rem;
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-primary);
    }

    .export-picker .control-toggle.disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .export-picker .control-toggle .muted.small {
      margin-left: 0.4rem;
    }

    .export-picker code {
      font-size: 0.75rem;
    }

    .link-button {
      background: none;
      border: none;
      padding: 0;
      color: var(--action-bg);
      cursor: pointer;
      text-decoration: underline;
      font: inherit;
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
      /* Same min-width:0 trick — without it, any nested grid child with a
         wide intrinsic size (long path string, native file input chrome)
         forces the wrapper wider than the splitter panel. */
      min-width: 0;
    }

    .wizard-current-page {
      min-width: 0;
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
  protected readonly wizardPageOptional = WIZARD_PAGE_OPTIONAL;
  protected readonly stageSummaries = WIZARD_STAGE_SUMMARIES;
  protected readonly maskBackends = WIZARD_MASK_BACKENDS;
  protected readonly sessionTree = inject(SessionTreeService);
  protected readonly sessionService = inject(SessionService);
  protected readonly sculptokService = inject(SculptokService);
  protected readonly planService = inject(PlanService);
  protected readonly renderService = inject(RenderService);
  protected readonly exportService = inject(ExportService);
  private readonly apiClient = inject(ApiClientService);
  private readonly maskService = inject(MaskService);
  protected readonly session = this.sessionTree.session;
  protected readonly pipeline = this.sessionTree.pipeline;
  protected readonly output = this.sessionTree.output;
  protected readonly ui = this.sessionTree.ui;
  protected readonly toasts = computed(() => this.ui().toasts);
  protected readonly splitterSizes = computed(() => this.ui().rightPaneCollapsed
    ? [...WIZARD_COLLAPSED_SPLITTER_SIZES]
    : [...WIZARD_DEFAULT_SPLITTER_SIZES]);
  protected readonly historyPreview = computed(() => this.session().history.slice(0, WIZARD_HISTORY_PREVIEW_LIMIT));

  /**
   * Tracks the (image, heightmap, profile) tuple the auto-plan effect last
   * dispatched a compute for. Without this the effect would loop, since
   * computePlan() writes to output.plan which the effect reads. Held as a
   * signal so the autoPlanAttempted computed below stays reactive.
   */
  private readonly lastAutoPlanKey = signal<string | null>(null);

  /**
   * User selection for the Submit-the-bundle action on step 5. All three
   * default true; when a prerequisite isn't met the checkbox is disabled
   * and its effective value drops out of the bundle request.
   */
  protected readonly exportSelections = signal({ png: true, lbrn2: true, stl: true });

  /**
   * Wall-clock signal that ticks once per second so ``relativeTime`` is
   * deterministic within a change-detection cycle. Without this, calling
   * ``Date.now()`` inline would flip "41s ago" to "42s ago" between
   * Angular's two CD passes and trigger NG0100. The interval lives on
   * the component so it's torn down on destroy (DestroyRef hook below).
   */
  private readonly nowMs = signal(Date.now());

  /**
   * Exposes "the auto-effect already fired for the current key" to the
   * template so the disabled-hint can distinguish "compute is queued" from
   * "compute already ran and didn't produce a plan — retry manually".
   */
  protected readonly autoPlanAttempted = computed(() => {
    const session = this.session();
    const heightmapId = this.output().heightmapId;
    const profileName = this.pipeline().render.profileName;
    if (!session.imageId || !heightmapId || !profileName) return false;
    const key = `${session.imageId}|${heightmapId}|${profileName}`;
    return this.lastAutoPlanKey() === key;
  });

  constructor() {
    this.sessionService.loadProfiles();
    this.sculptokService.loadCredits();

    // Tick the relative-time clock once per second — only in the browser,
    // and tear down on destroy so the test harness doesn't leak intervals.
    const platformId = inject(PLATFORM_ID);
    if (isPlatformBrowser(platformId)) {
      const handle = globalThis.setInterval(() => this.nowMs.set(Date.now()), 1000);
      inject(DestroyRef).onDestroy(() => globalThis.clearInterval(handle));
    }

    // Auto-compute pass plan whenever prerequisites are ready — independent
    // of which page the user is on. If they skip page 4 and jump straight
    // to Review, the plan still computes in the background so .lbrn2 export
    // is ready when they get there.
    effect(() => {
      const session = this.session();
      const heightmapId = this.output().heightmapId;
      const profileName = this.pipeline().render.profileName;
      if (!session.imageId || !heightmapId || !profileName) return;
      const key = `${session.imageId}|${heightmapId}|${profileName}`;
      if (this.lastAutoPlanKey() === key) return;
      if (this.output().plan) return; // existing plan is already current; render/profile changes clear it
      if (this.planService.inFlight()) return;
      this.lastAutoPlanKey.set(key);
      this.planService.computePlan();
    });
  }

  /**
   * Returns whether a wizard step has produced its required artifact.
   * Used by the chip indicator to show ✓/○ at a glance.
   *
   * Step semantics:
   *   0 Upload          — complete when an image is uploaded
   *   1 Subject (mask)  — optional; complete when a mask exists
   *   2 Prep & Refine   — complete when the heightmap has rendered
   *   3 Material/Passes — complete when a pass plan exists
   *   4 Review & Export — terminal; "complete" once a heightmap is ready
   */
  protected pageStatus(index: number): 'complete' | 'incomplete' {
    switch (index) {
      case 0: return this.session().imageId ? 'complete' : 'incomplete';
      case 1: return this.pipeline().mask.maskId ? 'complete' : 'incomplete';
      case 2: return this.output().heightmapId ? 'complete' : 'incomplete';
      case 3: return this.output().plan ? 'complete' : 'incomplete';
      case 4: return this.output().heightmapId ? 'complete' : 'incomplete';
      default: return 'incomplete';
    }
  }

  protected pageStatusIcon(index: number): string {
    if (this.pageStatus(index) === 'complete') return '✓';
    return WIZARD_PAGE_OPTIONAL[index] ? '·' : '○';
  }

  protected pageStatusLabel(index: number): string {
    if (this.pageStatus(index) === 'complete') return 'complete';
    return WIZARD_PAGE_OPTIONAL[index] ? 'optional, not done' : 'not yet done';
  }

  protected blobUrl(id: string): string {
    return this.apiClient.blobUrl(id);
  }

  protected relativeTime(iso: string | null | undefined): string {
    if (!iso) return '—';
    const t = new Date(iso).getTime();
    if (Number.isNaN(t)) return '—';
    // Read from the per-second signal, NOT Date.now(), so the same value
    // is returned across both passes of dev-mode change detection.
    const seconds = Math.max(0, Math.round((this.nowMs() - t) / 1000));
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

  protected sculptokGenerate(): void {
    // Auto-enable subject mask when a bg-replace pattern is selected.
    // The composite step would silently no-op without a mask, leaving
    // the user wondering why their bg-replace had no effect.
    if (
      this.pipeline().settings.background_pattern !== 'none' &&
      !this.pipeline().settings.subject_mask_enabled
    ) {
      this.renderService.patchSettings('subject_mask_enabled', true);
    }
    this.sculptokService.generate();
  }

  protected onBackgroundPatternChange(event: Event): void {
    const value = (event.target as HTMLSelectElement).value as
      | 'none' | 'solid_black' | 'solid_white' | 'solid_grey'
      | 'guilloche' | 'stripes' | 'dots' | 'halftone' | 'checkers';
    this.renderService.patchSettings('background_pattern', value);
    if (value !== 'none' && !this.pipeline().settings.subject_mask_enabled) {
      this.renderService.patchSettings('subject_mask_enabled', true);
    }
  }

  protected onHeightmapFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.item(0);
    if (!file) return;
    this.sculptokService.uploadHeightmap(file);
    input.value = '';
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

  /** Effective selection — checkbox state AND its prerequisite. Used by both
   * the submit button label and the bundle request. */
  private effectiveSelections(): { png: boolean; lbrn2: boolean; stl: boolean } {
    const sel = this.exportSelections();
    const hm = !!this.output().heightmapId;
    const plan = !!this.output().plan;
    return { png: sel.png && hm, lbrn2: sel.lbrn2 && plan, stl: sel.stl && hm };
  }

  protected toggleExport(format: 'png' | 'lbrn2' | 'stl', event: Event): void {
    const checked = (event.target as HTMLInputElement).checked;
    this.exportSelections.update((s) => ({ ...s, [format]: checked }));
  }

  protected canSubmitBundle(): boolean {
    const eff = this.effectiveSelections();
    return eff.png || eff.lbrn2 || eff.stl;
  }

  protected submitButtonSummary(): string {
    const eff = this.effectiveSelections();
    const parts: string[] = [];
    if (eff.png) parts.push('PNG');
    if (eff.lbrn2) parts.push('.lbrn2');
    if (eff.stl) parts.push('.stl');
    return parts.length ? parts.join(' + ') : '(nothing selected)';
  }

  protected submitBundle(): void {
    this.exportService.exportBundle(this.effectiveSelections());
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

