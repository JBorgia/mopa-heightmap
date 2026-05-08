/**
 * Credits badge — compact nav-bar widget showing the user's remaining credit
 * balance with a "Get more" call to action.
 *
 * Source-of-truth precedence:
 *   1. ``AuthService.creditsRemaining()`` when signed in (SaaS tenant).
 *   2. ``SculptokService.credits()?.balance`` as a local-dev fallback so the
 *      widget still renders something useful when running against a
 *      developer-mode backend.
 *
 * Severity colors swap on the same thresholds as the existing tone tokens:
 *   * ``balance <= 0`` → red (danger)
 *   * ``balance <= 3`` → amber (warning)
 *   * otherwise         → muted/neutral
 */
import { CommonModule } from '@angular/common';
import { Component, computed, inject } from '@angular/core';

import { SculptokService } from '../state/sculptok.service';
import { AuthService } from '../saas/auth.service';

@Component({
  selector: 'app-credits-badge',
  standalone: true,
  imports: [CommonModule],
  template: `
    <span class="credits-badge"
          [class.warn]="severity() === 'warn'"
          [class.danger]="severity() === 'danger'">
      <span class="dot" aria-hidden="true"></span>
      <span class="count">{{ balance() }} credits</span>
    </span>
    <button type="button" class="topup" (click)="onGetMore()">Get more</button>
  `,
  styles: `
    :host {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
    }
    .credits-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      padding: 0.2rem 0.55rem;
      border: 1px solid var(--border-input);
      border-radius: 999px;
      background: var(--bg-sunken);
      color: var(--text-muted);
      font-size: 0.78rem;
      font-weight: 500;
      line-height: 1;
    }
    .dot {
      width: 0.45rem;
      height: 0.45rem;
      border-radius: 999px;
      background: currentColor;
      opacity: 0.7;
    }
    .count { color: var(--text-primary); }
    .credits-badge.warn  { color: #c98a18; border-color: #c98a18; }
    .credits-badge.warn  .count { color: #c98a18; }
    .credits-badge.danger { color: #d23a3a; border-color: #d23a3a; }
    .credits-badge.danger .count { color: #d23a3a; }
    .topup {
      padding: 0.2rem 0.6rem;
      border-radius: 0.4rem;
      border: 1px solid transparent;
      background: var(--action-bg);
      color: var(--text-primary);
      font-size: 0.78rem;
      cursor: pointer;
    }
    .topup:hover, .topup:focus-visible { filter: brightness(1.08); outline: none; }
  `,
})
export class CreditsBadgeComponent {
  private readonly auth = inject(AuthService);
  private readonly sculptok = inject(SculptokService);

  protected readonly balance = computed<number>(() =>
    this.auth.isAuthenticated()
      ? this.auth.creditsRemaining()
      : this.sculptok.credits()?.balance ?? 0,
  );

  protected readonly severity = computed<'normal' | 'warn' | 'danger'>(() => {
    const n = this.balance();
    if (n <= 0) return 'danger';
    if (n <= 3) return 'warn';
    return 'normal';
  });

  protected onGetMore(): void {
    // TODO: wire to Polar checkout once the SaaS backend is online.
  }
}
