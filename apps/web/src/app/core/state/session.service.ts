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
          },
          output: {
            ...current.output,
            heightmapId: null,
            previewId: null,
            plan: null,
            elapsedSeconds: null,
          },
        }));
        this.sessionTree.pushHistory(`session:upload:${file.name}`);
      },
      complete: () => {
        this.uploadInFlight.set(false);
      },
      error: () => {
        this.uploadInFlight.set(false);
      },
    });
  }

  setProfileName(profileName: string | null): void {
    this.sessionTree.patchState((current) => ({
      ...current,
      pipeline: {
        ...current.pipeline,
        render: {
          ...current.pipeline.render,
          profileName,
        },
      },
    }));
    this.sessionTree.pushHistory(`profile:select:${profileName ?? 'none'}`);
  }
}