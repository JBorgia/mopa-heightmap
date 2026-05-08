import {
  Component,
  HostListener,
  inject,
  signal,
} from '@angular/core';
import { CommonModule } from '@angular/common';

import { AuthService } from '../saas/auth.service';

@Component({
  selector: 'app-auth-modal',
  standalone: true,
  imports: [CommonModule],
  template: `
    @if (authService.authModalOpen()) {
      <div class="auth-backdrop" (click)="close()" aria-hidden="true"></div>
      <div
        class="auth-modal"
        role="dialog"
        aria-modal="true"
        aria-label="Sign in to MOPA Heightmap Studio"
      >
        <button type="button" class="auth-close" (click)="close()" aria-label="Close">×</button>

        <div class="auth-brand">
          <strong>MOPA Heightmap Studio</strong>
        </div>

        <div class="auth-tabs" role="tablist">
          <button
            type="button"
            role="tab"
            [class.active]="mode() === 'signin'"
            [attr.aria-selected]="mode() === 'signin'"
            (click)="mode.set('signin')"
          >Sign in</button>
          <button
            type="button"
            role="tab"
            [class.active]="mode() === 'signup'"
            [attr.aria-selected]="mode() === 'signup'"
            (click)="mode.set('signup')"
          >Create account</button>
        </div>

        <form class="auth-form" (submit)="submit($event)" novalidate>
          <div class="auth-field">
            <label for="auth-email">Email</label>
            <input
              id="auth-email"
              type="email"
              autocomplete="email"
              placeholder="you@example.com"
              [value]="email()"
              (input)="email.set($any($event.target).value)"
              required
              autofocus
            />
          </div>
          <div class="auth-field">
            <label for="auth-password">Password</label>
            <input
              id="auth-password"
              type="password"
              [attr.autocomplete]="mode() === 'signin' ? 'current-password' : 'new-password'"
              placeholder="{{ mode() === 'signup' ? 'At least 8 characters' : '' }}"
              [value]="password()"
              (input)="password.set($any($event.target).value)"
              required
            />
          </div>

          @if (error()) {
            <p class="auth-error" role="alert">{{ error() }}</p>
          }

          <button type="submit" class="auth-submit" [disabled]="loading()">
            @if (loading()) {
              Working…
            } @else if (mode() === 'signin') {
              Sign in →
            } @else {
              Create account →
            }
          </button>
        </form>

        <p class="auth-switch">
          @if (mode() === 'signin') {
            No account?
            <button type="button" class="link-btn" (click)="mode.set('signup')">Create one free →</button>
          } @else {
            Already have one?
            <button type="button" class="link-btn" (click)="mode.set('signin')">Sign in →</button>
          }
        </p>

        <p class="auth-legal">
          By signing in you agree to our Terms of Service. Payments handled by
          <a href="https://polar.sh" target="_blank" rel="noopener">Polar.sh</a>.
        </p>
      </div>
    }
  `,
  styles: `
    .auth-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.45);
      backdrop-filter: blur(2px);
      z-index: 9100;
    }

    .auth-modal {
      position: fixed;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      width: min(420px, calc(100vw - 2rem));
      z-index: 9101;
      background: var(--bg-surface);
      border: 1px solid var(--border-default);
      border-radius: 1.25rem;
      box-shadow: 0 24px 48px rgba(0,0,0,0.24);
      padding: 2rem;
      display: grid;
      gap: 1.25rem;
    }

    .auth-close {
      position: absolute;
      top: 1rem;
      right: 1rem;
      border: none;
      background: transparent;
      font-size: 1.4rem;
      line-height: 1;
      color: var(--text-muted);
      cursor: pointer;
      padding: 0.1rem 0.3rem;
      border-radius: 0.25rem;
    }

    .auth-close:hover { color: var(--text-primary); }

    .auth-brand strong {
      font-size: 0.875rem;
      color: var(--text-muted);
      letter-spacing: 0.03em;
    }

    .auth-tabs {
      display: flex;
      border-bottom: 1px solid var(--border-default);
      gap: 0;
    }

    .auth-tabs button {
      flex: 1;
      border: none;
      border-bottom: 2px solid transparent;
      background: transparent;
      font: inherit;
      font-size: 0.9rem;
      font-weight: 600;
      color: var(--text-muted);
      padding: 0.6rem 0;
      cursor: pointer;
      margin-bottom: -1px;
      transition: color 120ms, border-color 120ms;
    }

    .auth-tabs button.active {
      color: var(--text-primary);
      border-bottom-color: var(--action-bg);
    }

    .auth-form {
      display: grid;
      gap: 1rem;
    }

    .auth-field {
      display: grid;
      gap: 0.35rem;
    }

    .auth-field label {
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-primary);
    }

    .auth-field input {
      border: 1px solid var(--border-input);
      border-radius: 0.6rem;
      background: var(--bg-input);
      color: var(--text-primary);
      font: inherit;
      font-size: 0.9rem;
      padding: 0.65rem 0.85rem;
      outline: none;
      transition: border-color 120ms;
    }

    .auth-field input:focus { border-color: var(--action-bg); }

    .auth-error {
      font-size: 0.85rem;
      color: #d23a3a;
      margin: 0;
      padding: 0.5rem 0.75rem;
      background: color-mix(in srgb, #d23a3a 8%, var(--bg-surface));
      border: 1px solid color-mix(in srgb, #d23a3a 30%, var(--border-default));
      border-radius: 0.5rem;
    }

    .auth-submit {
      width: 100%;
      border-radius: 999px;
      border: none;
      background: var(--action-bg);
      color: var(--action-fg);
      font: inherit;
      font-size: 0.95rem;
      font-weight: 700;
      padding: 0.7rem 1rem;
      cursor: pointer;
      transition: filter 120ms;
    }

    .auth-submit:hover:not(:disabled) { filter: brightness(1.08); }
    .auth-submit:disabled { opacity: 0.5; cursor: not-allowed; }

    .auth-switch {
      margin: 0;
      font-size: 0.85rem;
      color: var(--text-muted);
      text-align: center;
    }

    .link-btn {
      border: none;
      background: transparent;
      color: var(--action-bg);
      font: inherit;
      font-size: inherit;
      cursor: pointer;
      text-decoration: underline;
      padding: 0;
    }

    .auth-legal {
      margin: 0;
      font-size: 0.72rem;
      color: var(--text-faint);
      text-align: center;
      line-height: 1.5;
    }

    .auth-legal a { color: var(--text-muted); }
  `,
})
export class AuthModalComponent {
  protected readonly authService = inject(AuthService);

  protected readonly mode = signal<'signin' | 'signup'>('signin');
  protected readonly email = signal('');
  protected readonly password = signal('');
  protected readonly loading = signal(false);
  protected readonly error = signal('');

  @HostListener('document:keydown', ['$event'])
  onKeydown(event: KeyboardEvent): void {
    if (event.key === 'Escape' && this.authService.authModalOpen()) {
      this.close();
    }
  }

  protected close(): void {
    this.authService.authModalOpen.set(false);
    this.error.set('');
  }

  protected async submit(event: Event): Promise<void> {
    event.preventDefault();
    this.error.set('');
    this.loading.set(true);
    try {
      if (this.mode() === 'signin') {
        await this.authService.signIn(this.email(), this.password());
      } else {
        await this.authService.signUp(this.email(), this.password());
      }
      this.close();
    } catch (err: unknown) {
      this.error.set(err instanceof Error ? err.message : 'Something went wrong — please try again.');
    } finally {
      this.loading.set(false);
    }
  }
}
