/**
 * Info-tip — small "ⓘ" trigger that reveals a definition.
 *
 * Two interactions, both supported:
 *   * Hover (mouse): the popover appears on `:hover`/`:focus-visible`.
 *   * Click / Enter: pins the popover open so touch users and anyone who
 *     wants to keep reading can do so. Click outside or hit Escape to close.
 *
 * Accessible: the trigger is a real `<button>` with aria-expanded + aria-
 * controls; the popover gets role="tooltip" and an aria-labelledby link.
 *
 * Drop-in usage:
 *   <app-info-tip text="CLAHE = local contrast boost..."></app-info-tip>
 *
 * Keep the text short — anything book-length belongs in the docs site.
 */
import { CommonModule } from '@angular/common';
import { Component, ElementRef, HostListener, Input, inject, signal } from '@angular/core';

@Component({
  selector: 'app-info-tip',
  standalone: true,
  imports: [CommonModule],
  template: `
    <span class="info-tip-wrapper">
      <button
        type="button"
        class="info-tip-trigger"
        [attr.aria-label]="'More info' + (label ? ': ' + label : '')"
        [attr.aria-expanded]="pinned()"
        (click)="onToggle($event)"
        (keydown.escape)="pinned.set(false)"
      >
        <span aria-hidden="true">i</span>
      </button>
      <span class="info-tip-popover"
            role="tooltip"
            [class.pinned]="pinned()">
        {{ text }}
      </span>
    </span>
  `,
  styles: `
    .info-tip-wrapper {
      position: relative;
      display: inline-flex;
      vertical-align: middle;
      margin-left: 0.35rem;
    }

    .info-tip-trigger {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 1.05rem;
      height: 1.05rem;
      padding: 0;
      border-radius: 999px;
      border: 1px solid var(--border-input);
      background: var(--bg-sunken);
      color: var(--text-muted);
      font-family: serif;
      font-style: italic;
      font-weight: 700;
      font-size: 0.7rem;
      line-height: 1;
      cursor: help;
      flex-shrink: 0;
    }

    .info-tip-trigger:hover,
    .info-tip-trigger:focus-visible {
      background: var(--action-bg);
      color: var(--action-fg);
      border-color: var(--action-bg);
      outline: none;
    }

    .info-tip-popover {
      position: absolute;
      bottom: calc(100% + 0.4rem);
      left: 50%;
      transform: translateX(-50%);
      min-width: 14rem;
      max-width: 20rem;
      padding: 0.55rem 0.75rem;
      border: 1px solid var(--border-default);
      border-radius: 0.5rem;
      background: var(--bg-surface);
      color: var(--text-primary);
      font-size: 0.8rem;
      font-weight: 400;
      line-height: 1.35;
      text-align: left;
      letter-spacing: normal;
      text-transform: none;
      box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25);
      z-index: 100;
      pointer-events: none;
      opacity: 0;
      transform-origin: bottom center;
      transition: opacity 80ms ease;
      white-space: normal;
      overflow-wrap: anywhere;
    }

    .info-tip-wrapper:hover .info-tip-popover,
    .info-tip-trigger:focus-visible + .info-tip-popover,
    .info-tip-popover.pinned {
      opacity: 1;
      pointer-events: auto;
    }

    /* Keep the popover readable on narrow viewports (right-pane cells). */
    @media (max-width: 720px) {
      .info-tip-popover {
        left: auto;
        right: 0;
        transform: none;
        max-width: min(18rem, calc(100vw - 2rem));
      }
    }
  `,
})
export class InfoTipComponent {
  /** The definition / explanation. Required. */
  @Input({ required: true }) text!: string;

  /** Optional control name — used to enrich the aria-label. */
  @Input() label?: string;

  protected readonly pinned = signal(false);
  private readonly hostElement = inject(ElementRef<HTMLElement>);

  protected onToggle(event: MouseEvent): void {
    event.stopPropagation();
    this.pinned.update((v) => !v);
  }

  /** Close any pinned popover when the user clicks elsewhere. */
  @HostListener('document:click', ['$event'])
  protected onDocumentClick(event: MouseEvent): void {
    if (!this.pinned()) return;
    const target = event.target as Node | null;
    if (!target || !this.hostElement.nativeElement.contains(target)) {
      this.pinned.set(false);
    }
  }
}
