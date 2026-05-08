import { Injectable, computed, signal } from '@angular/core';
import type { User as SupabaseUser } from '@supabase/supabase-js';

import { supabase, supabaseConfigured } from './supabase';
import type { PricingTier } from './pricing';

export interface UserProfile {
  id: string;
  email: string;
  tier: PricingTier;
  creditsRemaining: number;
}

@Injectable({ providedIn: 'root' })
export class AuthService {
  readonly currentUser = signal<UserProfile | null>(null);
  readonly isAuthenticated = computed(() => this.currentUser() !== null);
  readonly tier = computed<PricingTier>(() => this.currentUser()?.tier ?? 'free');
  readonly creditsRemaining = computed(() => this.currentUser()?.creditsRemaining ?? 0);

  /** Controls the auth modal — set true from any component to open it. */
  readonly authModalOpen = signal(false);

  async signIn(email: string, password: string): Promise<void> {
    if (!supabaseConfigured) throw new Error('Supabase not configured — set meta tags in index.html.');
    const { data, error } = await supabase.auth.signInWithPassword({ email, password });
    if (error) throw error;
    if (data.user) this.currentUser.set(this._mapUser(data.user));
  }

  async signUp(email: string, password: string): Promise<void> {
    if (!supabaseConfigured) throw new Error('Supabase not configured — set meta tags in index.html.');
    const { data, error } = await supabase.auth.signUp({ email, password });
    if (error) throw error;
    if (data.user) this.currentUser.set(this._mapUser(data.user));
  }

  async signOut(): Promise<void> {
    if (supabaseConfigured) await supabase.auth.signOut();
    globalThis.localStorage?.removeItem('mopa-dev-user');
    this.currentUser.set(null);
  }

  async initialize(): Promise<void> {
    if (supabaseConfigured) {
      const { data: { session } } = await supabase.auth.getSession();
      if (session?.user) {
        this.currentUser.set(this._mapUser(session.user));
        supabase.auth.onAuthStateChange((_event, s) => {
          this.currentUser.set(s?.user ? this._mapUser(s.user) : null);
        });
        return;
      }
    }
    // Local-dev fallback
    const stored = globalThis.localStorage?.getItem('mopa-dev-user');
    if (stored) {
      try { this.currentUser.set(JSON.parse(stored) as UserProfile); } catch { /* ignore */ }
    }
  }

  private _mapUser(user: SupabaseUser): UserProfile {
    const meta = (user.user_metadata ?? {}) as Partial<UserProfile>;
    return {
      id: user.id,
      email: user.email ?? '',
      tier: (meta.tier as PricingTier) ?? 'free',
      creditsRemaining: meta.creditsRemaining ?? 3,
    };
  }
}
