import { Component } from '@angular/core';
import { Card } from 'primeng/card';

@Component({
  selector: 'app-wizard',
  standalone: true,
  imports: [Card],
  template: `
    <div class="flex justify-content-center align-items-center" style="min-height: 80vh">
      <p-card header="Wizard" subheader="Heightmap Wizard" [style]="{'width': '480px'}">
        <p>Wizard UI coming soon (Phase 9d).</p>
      </p-card>
    </div>
  `,
})
export class WizardComponent {}
