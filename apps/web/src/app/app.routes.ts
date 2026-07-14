import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', redirectTo: '/wizard', pathMatch: 'full' },
  {
    path: 'wizard',
    loadComponent: () => import('./features/wizard/wizard-shell.component').then(m => m.WizardShellComponent),
  },
  {
    path: 'pricing',
    loadComponent: () => import('./features/pricing/pricing-page.component').then(m => m.PricingPageComponent),
  },
  { path: '**', redirectTo: '/wizard' },
];
