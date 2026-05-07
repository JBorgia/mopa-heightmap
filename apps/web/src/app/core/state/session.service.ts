import { Injectable, inject, signal } from '@angular/core';

import { ApiClientService } from '../api/api-client.service';
import { ProfileSummary } from '../api/api-types';
import { SessionTreeService } from './session-tree.service';

@Injectable({ providedIn: 'root' })
export class SessionService {
  private readonly apiClient = inject(ApiClientService);
  private readonly sessionTree = inject(SessionTreeService);

  readonly profiles = signal<ProfileSummary[]>([]);
  readonly profilesLoaded = signal(false);
  readonly uploadInFlight = signal(false);

  loadProfiles(): void {
    if (this.profilesLoaded()) {
      return;
    }

    this.apiClient.listProfiles().subscribe((profiles) => {
      this.profiles.set(profiles);
      this.profilesLoaded.set(true);

      const currentProfileName = this.sessionTree.pipeline().render.profileName;
      if (!currentProfileName && profiles.length > 0) {
        this.setProfileName(profiles[0].name);
      }
    });
  }

  uploadImage(file: File): void {
    this.uploadInFlight.set(true);
    this.apiClient.uploadImage(file).subscribe({
      next: (response) => {
        this.sessionTree.patchState((current) => ({
          ...current,
          session: {
            ...current.session,
            imageId: response.image_id,
            imageHash: response.sha256,
            sourceMeta: {
              w: response.w,
              h: response.h,
              bytes: file.size,
            },
          },
          pipeline: {
            ...current.pipeline,
            mask: {
              ...current.pipeline.mask,
              maskId: null,
              coveragePct: 0,
            },
            // The previous heightmap was either sculptok-generated FROM the
            // previous photo (so it's irrelevant to this one) or a hand-
            // supplied PNG the user picked for that subject. Either way,
            // carrying it over and silently rendering with it would be
            // wrong. Force the user to re-pick on step 3.
            settings: {
              ...current.pipeline.settings,
              external_heightmap_path: '',
            },
          },
          output: {
            ...current.output,
            heightmapId: null,
            previewId: null,
            conditionedId: null,
            sculptokInputId: null,
            renderMaskId: null,
            plan: null,
            elapsedSeconds: null,
          },
        }));
        this.sessionTree.pushHistory(`session:upload:${file.name}`);
      },
      complete: () => {
        this.uploadInFlight.set(false);
      },
      error: (err) => {
        const detail = err?.error?.detail ?? err?.message ?? 'Unknown error';
        this.sessionTree.addToast({
          id: crypto.randomUUID(),
          severity: 'error',
          summary: 'Upload failed',
          detail,
        });
        this.uploadInFlight.set(false);
      },
    });
  }

  setProfileName(profileName: string | null): void {
    this.sessionTree.patchState((current) => {
      const profileChanged = current.pipeline.render.profileName !== profileName;
      return {
        ...current,
        pipeline: {
          ...current.pipeline,
          render: {
            ...current.pipeline.render,
            profileName,
          },
        },
        // The pass plan is keyed to (heightmap, profile, settings); a new
        // profile invalidates it. Clear so the wizard's auto-compute will
        // recompute against the new profile.
        output: profileChanged
          ? { ...current.output, plan: null }
          : current.output,
      };
    });
    this.sessionTree.pushHistory(`profile:select:${profileName ?? 'none'}`);
  }

  /** Persist the current ``pipeline.settings`` as a user-scope profile. */
  saveCurrentAsProfile(name: string, opts: { overwrite?: boolean } = {}): void {
    const settings = this.sessionTree.state().pipeline.settings;
    this.apiClient
      .saveProfile({
        name,
        settings,
        overwrite: opts.overwrite ?? false,
        machine: 'JPT MOPA fiber',
        lightburn_mode: '3D Sliced',
      })
      .subscribe({
        next: () => {
          // Force a refresh so the new profile appears in the dropdown.
          this.profilesLoaded.set(false);
          this.loadProfiles();
          this.setProfileName(name);
          this.sessionTree.pushHistory(`profile:save:${name}`);
          this.sessionTree.addToast({
            id: crypto.randomUUID(),
            severity: 'success',
            summary: 'Profile saved',
            detail: name,
          });
        },
        error: (err) => {
          const detail = err?.error?.detail ?? err?.message ?? 'Unknown error';
          this.sessionTree.addToast({
            id: crypto.randomUUID(),
            severity: 'error',
            summary: 'Save failed',
            detail,
          });
        },
      });
  }

  /** Delete a user-scope profile (shipped profiles are protected by the API). */
  deleteCurrentProfile(): void {
    const name = this.sessionTree.pipeline().render.profileName;
    if (!name) return;
    this.apiClient.deleteProfile(name).subscribe({
      next: () => {
        this.profilesLoaded.set(false);
        this.loadProfiles();
        this.setProfileName(null);
        this.sessionTree.pushHistory(`profile:delete:${name}`);
        this.sessionTree.addToast({
          id: crypto.randomUUID(),
          severity: 'success',
          summary: 'Profile deleted',
          detail: name,
        });
      },
      error: (err) => {
        const detail = err?.error?.detail ?? err?.message ?? 'Unknown error';
        this.sessionTree.addToast({
          id: crypto.randomUUID(),
          severity: 'error',
          summary: 'Delete failed',
          detail,
        });
      },
    });
  }
}