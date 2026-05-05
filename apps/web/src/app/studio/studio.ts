import { Component } from '@angular/core';
import { Card } from 'primeng/card';

@Component({
  selector: 'app-studio',
  standalone: true,
  imports: [Card],
  template: `
    <div class="flex justify-content-center align-items-center" style="min-height: 80vh">
      <p-card header="Studio" subheader="Heightmap Studio" [style]="{'width': '480px'}">
        <p>Studio UI coming soon (Phase 9e).</p>
      </p-card>
    </div>
  `,
})
export class StudioComponent {}
