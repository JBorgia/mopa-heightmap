/**
 * Pricing tier definitions for the MOPA Heightmap SaaS.
 *
 * Pure TypeScript — no Angular dependencies — so this module can be imported
 * by services, components, marketing pages, and (eventually) server-side
 * checkout code without dragging the framework along.
 *
 * Pricing model summary:
 *   * free   — 3 lifetime credits, no rollover, no recurring billing
 *   * maker  — recurring monthly/annual; 20 credits/mo with 2x rollover cap
 *   * shop   — recurring monthly/annual; 75 credits/mo with 2x rollover cap
 *
 * Top-up packs are sold separately — flat $12 for 10 credits — and stack on
 * top of whichever tier the user is on.
 */

export type PricingTier = 'free' | 'maker' | 'shop';

export interface TierDefinition {
  id: PricingTier;
  name: string;
  tagline: string;
  /** USD per month when billed monthly. 0 = free tier. */
  monthlyPrice: number;
  /** USD per year when billed annually (single charge). 0 = free tier. */
  annualPrice: number;
  /** Credits granted each billing period. 0 = lifetime credits (free tier). */
  creditsPerMonth: number;
  /** One-time credit allowance — only meaningful for the free tier. */
  lifetimeCredits: number;
  /**
   * Multiplier on ``creditsPerMonth`` capping how many credits a user may bank
   * via rollover. e.g. 2 = a maker user can hold up to 40 credits at once.
   */
  creditRolloverMultiplier: number;
  features: string[];
  /** When true, surface this tier as the "recommended" option in the UI. */
  highlight?: boolean;
}

export const PRICING_TIERS: readonly TierDefinition[] = [
  {
    id: 'free',
    name: 'Free',
    tagline: 'Try it out, no card required.',
    monthlyPrice: 0,
    annualPrice: 0,
    creditsPerMonth: 0,
    lifetimeCredits: 3,
    creditRolloverMultiplier: 0,
    features: [
      'Full quality output',
      'No watermark',
      'All export formats',
      'All material presets',
    ],
  },
  {
    id: 'maker',
    name: 'Maker',
    tagline: 'For hobbyists doing a few projects a week.',
    monthlyPrice: 9,
    annualPrice: 79,
    creditsPerMonth: 20,
    lifetimeCredits: 0,
    creditRolloverMultiplier: 2,
    highlight: true,
    features: [
      '20 projects/month',
      '2-month credit rollover',
      'All presets & profiles',
      '30-day download history',
    ],
  },
  {
    id: 'shop',
    name: 'Shop',
    tagline: 'For working shops and small studios.',
    monthlyPrice: 24,
    annualPrice: 199,
    creditsPerMonth: 75,
    lifetimeCredits: 0,
    creditRolloverMultiplier: 2,
    features: [
      '75 projects/month',
      'Priority processing',
      'Batch upload (5 at once)',
      'Unlimited download history',
      'API access',
      'Commercial use license',
      'Email support',
    ],
  },
] as const;

/** Top-up pack — flat $12 buys 10 extra credits, never expires. */
export const TOPUP_PRICE = 12;
export const TOPUP_CREDITS = 10;

/**
 * Return the percentage saved by paying annually rather than 12x the monthly
 * rate. Free tiers (price 0) return 0. Result is rounded to the nearest int.
 *
 * Example: maker tier — monthly $9, annual $79.
 *   12 * 9 = 108; saved = 108 - 79 = 29; 29 / 108 ≈ 26.85 → 27%.
 */
export function annualSavingsPercent(tier: TierDefinition): number {
  if (tier.monthlyPrice <= 0) return 0;
  const yearlyAtMonthly = tier.monthlyPrice * 12;
  if (yearlyAtMonthly <= 0) return 0;
  const saved = yearlyAtMonthly - tier.annualPrice;
  if (saved <= 0) return 0;
  return Math.round((saved / yearlyAtMonthly) * 100);
}

/**
 * Human-friendly framing of the annual discount expressed in "free months".
 *
 *   savedDollars = 12*monthly - annual
 *   freeMonths   = savedDollars / monthly  (rounded to the nearest integer,
 *                  with a half-month threshold producing "X.5 months free")
 *
 * Free tiers and tiers with no annual discount return an empty string so
 * callers can simply `*ngIf` on the result.
 */
export function annualFraming(tier: TierDefinition): string {
  if (tier.monthlyPrice <= 0) return '';
  const saved = tier.monthlyPrice * 12 - tier.annualPrice;
  if (saved <= 0) return '';
  const freeMonths = saved / tier.monthlyPrice;
  // Snap to nearest half so "2.97 months" presents as "3 months free".
  const snapped = Math.round(freeMonths * 2) / 2;
  if (snapped <= 0) return '';
  const label = Number.isInteger(snapped) ? `${snapped}` : snapped.toFixed(1);
  return `${label} months free`;
}
