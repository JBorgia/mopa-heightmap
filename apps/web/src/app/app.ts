import { Component, inject } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';

import { AuthModalComponent } from './core/ui/auth-modal.component';
import { CommandPaletteComponent } from './core/ui/command-palette.component';
import { CreditsBadgeComponent } from './core/ui/credits-badge.component';
import { AuthService } from './core/saas/auth.service';
import { SessionTreeService } from './core/state/session-tree.service';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, RouterLink, RouterLinkActive, CommandPaletteComponent, CreditsBadgeComponent, AuthModalComponent],
  templateUrl: './app.html',
  styleUrl: './app.scss'
})
export class App {
  protected readonly sessionTree = inject(SessionTreeService);
  protected readonly authService = inject(AuthService);

  constructor() {
    this.authService.initialize();
  }
}
