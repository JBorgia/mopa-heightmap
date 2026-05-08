import { Component, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Router } from '@angular/router';

import {
  PRICING_TIERS,
  TOPUP_PRICE,
  TOPUP_CREDITS,
  TierDefinition,
  PricingTier,
  annualSavingsPercent,
  annualFraming,
} from '../../core/saas/pricing';
import { AuthService } from '../../core/saas/auth.service';

@Component({
  selector: 'app-pricing-page',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="pricing-shell">

      <header class="pricing-header">
        <p class="eyebrow">Pricing</p>
        <h1>One credit. One production-ready LightBurn project.</h1>
        <p class="subhead">
          Upload a photo, pick your material, download a .lbrn2 ready to fire —
          zero LightBurn knowledge required.
        </p>
        <div class="billing-toggle" role="group" aria-label="Billing period">
          <button type="button"
            [class.active]="billing() === 'monthly'"
            (click)="billing.set('monthly')">
            Monthly
          </button>
          <button type="button"
            [class.active]="billing() === 'annual'"
            (click)="billing.set('annual')">
            Annual
            <span class="save-pill">Save up to {{ maxSavings }}%</span>
          </button>
        </div>
      </header>

      <div class="pricing-cards">
        @for (tier of pricingTiers; track tier.id) {
          <div class="pricing-card" [class.highlight]="tier.highlight">
            @if (tier.highlight) {
              <div class="popular-badge">Most popular</div>
            }
            <h2 class="tier-name">{{ tier.name }}</h2>
            <p class="tier-tagline">{{ tier.tagline }}</p>

            <div class="price-block">
              @if (tier.monthlyPrice === 0) {
                <span class="price-amount">Free</span>
                <span class="price-detail">{{ tier.lifetimeCredits }} lifetime credits — no card needed</span>
              } @else {
                <span class="price-amount">
                  {{ '$' + (billing() === 'annual' ? annualMonthly(tier) : tier.monthlyPrice) }}
                  <span class="price-per">/mo</span>
                </span>
                @if (billing() === 'annual') {
                  <span class="price-detail">
                    {{ '$' + tier.annualPrice }}/yr · {{ annualLabel(tier) }}
                  </span>
                } @else {
                  <span class="price-detail">billed monthly · cancel anytime</span>
                }
              }
            </div>

            <ul class="feature-list">
              @for (feat of tier.features; track feat) {
                <li><span class="check" aria-hidden="true">✓</span> {{ feat }}</li>
              }
            </ul>

            <button
              type="button"
              class="tier-cta"
              [class.tier-cta-primary]="tier.highlight"
              (click)="selectTier(tier.id)"
            >
              {{ tier.monthlyPrice === 0 ? 'Get started free' : 'Start ' + tier.name }}
            </button>
          </div>
        }
      </div>

      <section class="topup-section">
        <div class="topup-card">
          <div class="topup-text">
            <h3>Need a one-off top-up?</h3>
            <p>
              {{ topupCredits }} credits for {{ '$' + topupPrice }}. Never expires, stacks on any plan.
              Great for a batch of holiday gifts or a client rush job.
            </p>
          </div>
          <button type="button" class="topup-btn" (click)="buyTopup()">
            Buy {{ topupCredits }} credits — {{ '$' + topupPrice }}
          </button>
        </div>
      </section>

      <section class="faq-section">
        <h2>Common questions</h2>
        <dl class="faq-list">
          <div class="faq-item">
            <dt>What counts as one credit?</dt>
            <dd>One depth-map generation via the Sculptok AI API. Rendering, re-rendering, and exporting don't consume credits.</dd>
          </div>
          <div class="faq-item">
            <dt>Do unused credits roll over?</dt>
            <dd>On Maker and Shop, unused credits roll over up to 2× your monthly allowance. Free credits never expire.</dd>
          </div>
          <div class="faq-item">
            <dt>What laser machines does this work with?</dt>
            <dd>Any MOPA fiber laser that LightBurn supports — JPT, Raycus, Cloudray, and most OEM variants. The .lbrn2 output requires LightBurn 1.7 or newer.</dd>
          </div>
          <div class="faq-item">
            <dt>Can I upload my own depth maps?</dt>
            <dd>Yes — if you already have a greyscale PNG from Meshy, hand-painting, or any other tool, you can upload it directly and skip the Sculptok step entirely. No credit consumed.</dd>
          </div>
        </dl>
      </section>

    </div>
  `,
  styles: `
    .pricing-shell {
      max-width: 1100px;
      margin: 0 auto;
      padding: 3rem 1.5rem 5rem;
    }

    .pricing-header {
      text-align: center;
      margin-bottom: 3rem;
    }

    .eyebrow {
      text-transform: uppercase;
      letter-spacing: 0.1em;
      font-size: 0.75rem;
      font-weight: 700;
      color: var(--text-muted);
      margin: 0 0 0.75rem;
    }

    h1 {
      font-size: clamp(1.6rem, 3vw, 2.4rem);
      font-weight: 800;
      color: var(--text-primary);
      margin: 0 0 1rem;
      line-height: 1.2;
    }

    .subhead {
      font-size: 1rem;
      color: var(--text-secondary);
      max-width: 520px;
      margin: 0 auto 2rem;
      line-height: 1.6;
    }

    .billing-toggle {
      display: inline-flex;
      gap: 0;
      border: 1px solid var(--border-input);
      border-radius: 999px;
      background: var(--bg-sunken);
      padding: 0.2rem;
    }

    .billing-toggle button {
      border-radius: 999px;
      border: none;
      background: transparent;
      color: var(--text-muted);
      font: inherit;
      font-size: 0.875rem;
      font-weight: 600;
      padding: 0.4rem 1rem;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 0.5rem;
      transition: background 120ms, color 120ms;
    }

    .billing-toggle button.active {
      background: var(--bg-surface);
      color: var(--text-primary);
      box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    .save-pill {
      font-size: 0.68rem;
      font-weight: 700;
      background: #27ae60;
      color: #fff;
      border-radius: 999px;
      padding: 0.1rem 0.45rem;
      letter-spacing: 0.02em;
    }

    /* ── Cards ──────────────────────────────────────────────────────────── */
    .pricing-cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(270px, 1fr));
      gap: 1.25rem;
      align-items: start;
    }

    .pricing-card {
      position: relative;
      border: 1px solid var(--border-default);
      border-radius: 1.25rem;
      background: var(--bg-surface);
      padding: 1.75rem 1.5rem;
      display: grid;
      gap: 1.1rem;
    }

    .pricing-card.highlight {
      border-color: var(--action-bg);
      box-shadow: 0 0 0 1px var(--action-bg), 0 8px 24px rgba(0,0,0,0.1);
    }

    .popular-badge {
      position: absolute;
      top: -0.75rem;
      left: 50%;
      transform: translateX(-50%);
      background: var(--action-bg);
      color: var(--action-fg);
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      border-radius: 999px;
      padding: 0.25rem 0.85rem;
      white-space: nowrap;
    }

    .tier-name {
      font-size: 1.25rem;
      font-weight: 800;
      margin: 0;
      color: var(--text-primary);
    }

    .tier-tagline {
      font-size: 0.875rem;
      color: var(--text-muted);
      margin: 0;
      line-height: 1.5;
    }

    .price-block {
      display: grid;
      gap: 0.2rem;
    }

    .price-amount {
      font-size: 2.25rem;
      font-weight: 800;
      color: var(--text-primary);
      line-height: 1;
    }

    .price-per {
      font-size: 1rem;
      font-weight: 500;
      color: var(--text-muted);
    }

    .price-detail {
      font-size: 0.8rem;
      color: var(--text-muted);
    }

    .feature-list {
      list-style: none;
      padding: 0;
      margin: 0;
      display: grid;
      gap: 0.55rem;
      border-top: 1px solid var(--border-default);
      padding-top: 1rem;
    }

    .feature-list li {
      display: flex;
      align-items: baseline;
      gap: 0.5rem;
      font-size: 0.875rem;
      color: var(--text-secondary);
    }

    .check {
      color: #27ae60;
      font-weight: 700;
      flex-shrink: 0;
    }

    .tier-cta {
      width: 100%;
      border-radius: 999px;
      border: 1px solid var(--border-input);
      background: var(--bg-sunken);
      color: var(--text-primary);
      font: inherit;
      font-size: 0.9rem;
      font-weight: 700;
      padding: 0.65rem 1rem;
      cursor: pointer;
      transition: background 120ms, filter 120ms;
    }

    .tier-cta:hover { background: var(--bg-base); }

    .tier-cta-primary {
      background: var(--action-bg);
      color: var(--action-fg);
      border-color: var(--action-bg);
    }

    .tier-cta-primary:hover { filter: brightness(1.1); background: var(--action-bg); }

    /* ── Top-up ──────────────────────────────────────────────────────── */
    .topup-section {
      margin-top: 3rem;
    }

    .topup-card {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 2rem;
      flex-wrap: wrap;
      border: 1px solid var(--border-default);
      border-radius: 1rem;
      background: var(--bg-surface);
      padding: 1.5rem 2rem;
    }

    .topup-text h3 {
      margin: 0 0 0.4rem;
      font-size: 1rem;
      font-weight: 700;
      color: var(--text-primary);
    }

    .topup-text p {
      margin: 0;
      font-size: 0.875rem;
      color: var(--text-muted);
      max-width: 500px;
      line-height: 1.5;
    }

    .topup-btn {
      flex-shrink: 0;
      border-radius: 999px;
      border: 1px solid var(--border-input);
      background: var(--bg-sunken);
      color: var(--text-primary);
      font: inherit;
      font-size: 0.875rem;
      font-weight: 700;
      padding: 0.6rem 1.25rem;
      cursor: pointer;
      white-space: nowrap;
      transition: background 120ms;
    }

    .topup-btn:hover { background: var(--bg-base); }

    /* ── FAQ ─────────────────────────────────────────────────────────── */
    .faq-section {
      margin-top: 4rem;
    }

    .faq-section h2 {
      font-size: 1.3rem;
      font-weight: 700;
      color: var(--text-primary);
      margin: 0 0 1.5rem;
    }

    .faq-list {
      display: grid;
      gap: 1rem;
    }

    .faq-item {
      border: 1px solid var(--border-default);
      border-radius: 0.75rem;
      padding: 1rem 1.25rem;
      background: var(--bg-surface);
    }

    .faq-item dt {
      font-weight: 700;
      font-size: 0.9rem;
      color: var(--text-primary);
      margin-bottom: 0.4rem;
      text-transform: none;
      letter-spacing: 0;
    }

    .faq-item dd {
      margin: 0;
      font-size: 0.875rem;
      color: var(--text-secondary);
      line-height: 1.6;
      font-weight: 400;
    }
  `,
})
export class PricingPageComponent {
  private readonly router = inject(Router);
  private readonly authService = inject(AuthService);

  protected readonly billing = signal<'monthly' | 'annual'>('annual');
  protected readonly pricingTiers = PRICING_TIERS;
  protected readonly topupPrice = TOPUP_PRICE;
  protected readonly topupCredits = TOPUP_CREDITS;
  protected readonly maxSavings = Math.max(
    ...PRICING_TIERS.filter((t) => t.monthlyPrice > 0).map((t) => annualSavingsPercent(t)),
  );

  protected annualMonthly(tier: TierDefinition): string {
    return (tier.annualPrice / 12).toFixed(2).replace(/\.?0+$/, '');
  }

  protected annualLabel(tier: TierDefinition): string {
    return annualFraming(tier);
  }

  protected selectTier(id: PricingTier): void {
    if (id === 'free') {
      this.router.navigate(['/wizard']);
      return;
    }
    // TODO: open Polar checkout for the selected tier
    this.authService.authModalOpen.set(true);
  }

  protected buyTopup(): void {
    // TODO: open Polar checkout for the top-up pack
  }
}
