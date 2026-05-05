import { TestBed } from '@angular/core/testing';
import { describe, it, expect, beforeEach } from 'vitest';

import { BlobCacheService } from './blob-cache.service';
import { BLOB_CACHE_MAX_BYTES } from './studio-state';

function makeBlob(sizeBytes: number): Blob {
  return new Blob([new Uint8Array(sizeBytes)]);
}

describe('BlobCacheService', () => {
  let service: BlobCacheService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(BlobCacheService);
  });

  it('get returns null for unknown hash', () => {
    expect(service.get('no-such-hash')).toBeNull();
  });

  it('set then get returns the same blob', () => {
    const blob = makeBlob(100);
    service.set('hash-a', blob);
    expect(service.get('hash-a')).toBe(blob);
  });

  it('get updates lastAccessedAt (re-get does not throw)', () => {
    const blob = makeBlob(100);
    service.set('hash-b', blob);
    const first = service.get('hash-b');
    const second = service.get('hash-b');
    expect(first).toBe(second);
  });

  it('set with same hash replaces the entry', () => {
    const blobA = makeBlob(100);
    const blobB = makeBlob(200);
    service.set('hash-c', blobA);
    service.set('hash-c', blobB);
    expect(service.get('hash-c')).toBe(blobB);
  });

  it('clear removes all entries', () => {
    service.set('hash-d', makeBlob(100));
    service.set('hash-e', makeBlob(200));
    service.clear();
    expect(service.get('hash-d')).toBeNull();
    expect(service.get('hash-e')).toBeNull();
  });

  it('evicts least-recently-accessed entry when over budget', () => {
    // Fill the cache just over BLOB_CACHE_MAX_BYTES
    const halfBudget = Math.floor(BLOB_CACHE_MAX_BYTES / 2);
    const overBudget = halfBudget + 1;

    service.set('old', makeBlob(halfBudget));
    // Access 'old' so its lastAccessedAt is set, then set 'new' which is accessed later
    service.get('old');
    service.set('new', makeBlob(overBudget));

    // totalSize = halfBudget + overBudget > BLOB_CACHE_MAX_BYTES
    // 'old' was accessed first (earlier timestamp) → should be evicted
    expect(service.get('old')).toBeNull();
    expect(service.get('new')).not.toBeNull();
  });
});
