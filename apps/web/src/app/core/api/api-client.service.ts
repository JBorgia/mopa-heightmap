import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import {
  ClickMaskRequest,
  ExportLbrn2Request,
  ExportPngRequest,
  ExportStlRequest,
  MaskRequest,
  MaskResponse,
  PassPlanRequest,
  PassPlanResponse,
  ProfileSummary,
  RenderRequest,
  RenderResponse,
  UploadResponse,
} from './api-types';

export const API_BASE_URL = 'http://127.0.0.1:8000';

@Injectable({ providedIn: 'root' })
export class ApiClientService {
  private readonly httpClient = inject(HttpClient);

  uploadImage(file: File): Observable<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file, file.name);
    return this.httpClient.post<UploadResponse>(`${API_BASE_URL}/upload`, formData);
  }

  listProfiles(): Observable<ProfileSummary[]> {
    return this.httpClient.get<ProfileSummary[]>(`${API_BASE_URL}/profiles`);
  }

  createMask(request: MaskRequest): Observable<MaskResponse> {
    return this.httpClient.post<MaskResponse>(`${API_BASE_URL}/mask`, request);
  }

  clickMask(request: ClickMaskRequest): Observable<MaskResponse> {
    return this.httpClient.post<MaskResponse>(`${API_BASE_URL}/mask/click`, request);
  }

  render(request: RenderRequest): Observable<RenderResponse> {
    return this.httpClient.post<RenderResponse>(`${API_BASE_URL}/render`, request);
  }

  exportPng(request: ExportPngRequest): Observable<Blob> {
    return this.httpClient.post(`${API_BASE_URL}/export/png`, request, { responseType: 'blob' });
  }

  plan(request: PassPlanRequest): Observable<PassPlanResponse> {
    return this.httpClient.post<PassPlanResponse>(`${API_BASE_URL}/plan`, request);
  }

  exportLbrn2(request: ExportLbrn2Request): Observable<Blob> {
    return this.httpClient.post(`${API_BASE_URL}/export/lbrn2`, request, { responseType: 'blob' });
  }

  exportStl(request: ExportStlRequest): Observable<Blob> {
    return this.httpClient.post(`${API_BASE_URL}/export/stl`, request, { responseType: 'blob' });
  }

  blobUrl(blobId: string): string {
    return `${API_BASE_URL}/blob/${blobId}`;
  }
}