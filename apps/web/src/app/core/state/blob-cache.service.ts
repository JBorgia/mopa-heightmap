import { Injectable } from '@angular/core';

import { BLOB_CACHE_MAX_BYTES } from './studio-state';

interface BlobCacheEntry {
  blob: Blob;
  sizeBytes: number;
  lastAccessedAt: number;
}

@Injectable({ providedIn: 'root' })
export class BlobCacheService {
  private readonly entries = new Map<string, BlobCacheEntry>();
  private totalSizeBytes = 0;

  get(hash: string): Blob | null {
    const entry = this.entries.get(hash);
    if (!entry) {
      return null;
    }

    entry.lastAccessedAt = Date.now();
    return entry.blob;
  }

  set(hash: string, blob: Blob): void {
    const existing = this.entries.get(hash);
    if (existing) {
      this.totalSizeBytes -= existing.sizeBytes;
    }

    this.entries.set(hash, {
      blob,
      sizeBytes: blob.size,
      lastAccessedAt: Date.now(),
    });
    this.totalSizeBytes += blob.size;
    this.evictIfNeeded();
  }

  clear(): void {
    this.entries.clear();
    this.totalSizeBytes = 0;
  }

  private evictIfNeeded(): void {
    if (this.totalSizeBytes <= BLOB_CACHE_MAX_BYTES) {
      return;
    }

    const candidates = [...this.entries.entries()].sort((left, right) => left[1].lastAccessedAt - right[1].lastAccessedAt);
    for (const [hash, entry] of candidates) {
      this.entries.delete(hash);
      this.totalSizeBytes -= entry.sizeBytes;
      if (this.totalSizeBytes <= BLOB_CACHE_MAX_BYTES) {
        return;
      }
    }
  }
}