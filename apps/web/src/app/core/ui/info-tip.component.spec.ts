/**
 * Tests for InfoTipComponent — verify the click-to-pin and click-outside-
 * to-close behaviour. Hover-to-show is pure CSS so we don't test it here.
 */
import { Component } from '@angular/core';
import { TestBed } from '@angular/core/testing';
import { describe, it, expect, beforeEach } from 'vitest';

import { InfoTipComponent } from './info-tip.component';

@Component({
  standalone: true,
  imports: [InfoTipComponent],
  template: `
    <app-info-tip text="CLAHE = local contrast boost." label="CLAHE"></app-info-tip>
    <button type="button" id="other">Outside</button>
  `,
})
class HostComponent {}

describe('InfoTipComponent', () => {
  beforeEach(async () => {
    await TestBed.configureTestingModule({ imports: [HostComponent] }).compileComponents();
  });

  it('renders the trigger and the popover with the supplied text', () => {
    const fixture = TestBed.createComponent(HostComponent);
    fixture.detectChanges();
    const root = fixture.nativeElement as HTMLElement;
    const trigger = root.querySelector('.info-tip-trigger') as HTMLButtonElement;
    const popover = root.querySelector('.info-tip-popover') as HTMLElement;
    expect(trigger).toBeTruthy();
    expect(popover).toBeTruthy();
    expect(popover.textContent?.trim()).toBe('CLAHE = local contrast boost.');
  });

  it('aria-label includes the label prop', () => {
    const fixture = TestBed.createComponent(HostComponent);
    fixture.detectChanges();
    const root = fixture.nativeElement as HTMLElement;
    const trigger = root.querySelector('.info-tip-trigger') as HTMLButtonElement;
    expect(trigger.getAttribute('aria-label')).toBe('More info: CLAHE');
  });

  it('clicking the trigger pins the popover open (aria-expanded=true)', () => {
    const fixture = TestBed.createComponent(HostComponent);
    fixture.detectChanges();
    const root = fixture.nativeElement as HTMLElement;
    const trigger = root.querySelector('.info-tip-trigger') as HTMLButtonElement;
    trigger.click();
    fixture.detectChanges();
    expect(trigger.getAttribute('aria-expanded')).toBe('true');
    const popover = root.querySelector('.info-tip-popover') as HTMLElement;
    expect(popover.classList.contains('pinned')).toBe(true);
  });

  it('clicking the trigger again unpins it', () => {
    const fixture = TestBed.createComponent(HostComponent);
    fixture.detectChanges();
    const trigger = (fixture.nativeElement as HTMLElement).querySelector('.info-tip-trigger') as HTMLButtonElement;
    trigger.click();
    fixture.detectChanges();
    trigger.click();
    fixture.detectChanges();
    expect(trigger.getAttribute('aria-expanded')).toBe('false');
  });

  it('clicking outside the host unpins a pinned popover', () => {
    const fixture = TestBed.createComponent(HostComponent);
    fixture.detectChanges();
    const root = fixture.nativeElement as HTMLElement;
    const trigger = root.querySelector('.info-tip-trigger') as HTMLButtonElement;
    trigger.click();
    fixture.detectChanges();
    expect(trigger.getAttribute('aria-expanded')).toBe('true');

    const outside = root.querySelector('#other') as HTMLButtonElement;
    outside.click();
    fixture.detectChanges();
    expect(trigger.getAttribute('aria-expanded')).toBe('false');
  });
});
