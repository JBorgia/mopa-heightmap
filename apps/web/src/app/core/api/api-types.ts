import type { components } from './generated/api';

export type ProfileSummary = components['schemas']['ProfileSummary'];
export type UploadResponse = components['schemas']['UploadResponse'];
export type MaskRequest = components['schemas']['MaskRequest'];
export type MaskResponse = components['schemas']['MaskResponse'];
export type ClickMaskRequest = components['schemas']['ClickMaskRequest'];
export type RenderRequest = components['schemas']['RenderRequest'];
export type RenderResponse = components['schemas']['RenderResponse'];
export type ExportPngRequest = components['schemas']['ExportPngRequest'];
export type ExportLbrn2Request = components['schemas']['ExportLbrn2Request'];
export type ExportStlRequest = components['schemas']['ExportStlRequest'];
export type PassPlanRequest = components['schemas']['PassPlanRequest'];
export type PassPlanResponse = components['schemas']['PassPlanResponse'];
export type PassEntry = components['schemas']['PassEntry'];
export type ProfileDetail = components['schemas']['ProfileDetail'];
export type HeightmapSettings = components['schemas']['HeightmapSettings'];

export interface ApiErrorEnvelope {
  error: {
    code: string;
    message: string;
    hint?: string | null;
  };
}