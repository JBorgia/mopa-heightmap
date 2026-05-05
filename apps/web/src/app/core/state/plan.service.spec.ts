/**
 * Unit tests for PlanService.
 */
import { TestBed } from '@angular/core/testing';
import { of } from 'rxjs';
import { describe, it, expect, vi, beforeEach } from 'vitest';

import { ApiClientService } from '../api/api-client.service';
import { PlanService } from './plan.service';
import { SessionTreeService } from './session-tree.service';
import { DEFAULT_STUDIO_STATE } from './studio-state';

const PLAN_RESPONSE = {
  plan_id: 'plan-abc',
  passes: [
    { pass_number: 1, label: 'Form pass', depth_um: 50, color_hex: '#ff0000' },
    { pass_number: 2, label: 'Detail pass', depth_um: 20, color_hex: '#00ff00' },
  ],
};

function makeState(outputOverrides: Record<string, unknown> = {}, renderOverrides: Record<string, unknown> = {}) {
  return {
    ...DEFAULT_STUDIO_STATE,
    session: { ...DEFAULT_STUDIO_STATE.session, imageId: 'img-001' },
    output: { ...DEFAULT_STUDIO_STATE.output, heightmapId: 'hm-001', ...outputOverrides },
    pipeline: {
      ...DEFAULT_STUDIO_STATE.pipeline,
      render: { ...DEFAULT_STUDIO_STATE.pipeline.render, ...renderOverrides },
    },
  };
}

describe('PlanService', () => {
  let service: PlanService;
  let apiMock: { plan: ReturnType<typeof vi.fn> };
  let treeMock: {
    output: ReturnType<typeof vi.fn>;
    session: ReturnType<typeof vi.fn>;
    pipeline: ReturnType<typeof vi.fn>;
    patchState: ReturnType<typeof vi.fn>;
    pushHistory: ReturnType<typeof vi.fn>;
    _state: ReturnType<typeof makeState>;
  };

  beforeEach(() => {
    const state = makeState();
    apiMock = { plan: vi.fn(() => of(PLAN_RESPONSE)) };
    treeMock = {
      output: vi.fn(() => state.output),
      session: vi.fn(() => state.session),
      pipeline: vi.fn(() => state.pipeline),
      patchState: vi.fn(),
      pushHistory: vi.fn(),
      _state: state,
    };

    TestBed.configureTestingModule({
      providers: [
        PlanService,
        { provide: ApiClientService, useValue: apiMock },
        { provide: SessionTreeService, useValue: treeMock },
      ],
    });

    service = TestBed.inject(PlanService);
  });

  it('computePlan does nothing when heightmapId is null', () => {
    treeMock.output.mockReturnValue({ ...treeMock._state.output, heightmapId: null });
    service.computePlan();
    expect(apiMock.plan).not.toHaveBeenCalled();
  });

  it('computePlan does nothing when imageId is null', () => {
    treeMock.session.mockReturnValue({ ...treeMock._state.session, imageId: null });
    service.computePlan();
    expect(apiMock.plan).not.toHaveBeenCalled();
  });

  it('computePlan calls apiClient.plan with image_id and heightmap_id', () => {
    service.computePlan();
    expect(apiMock.plan).toHaveBeenCalledWith(
      expect.objectContaining({ image_id: 'img-001', heightmap_id: 'hm-001' }),
    );
  });

  it('computePlan patches output.plan with mapped passes', () => {
    service.computePlan();
    const patcher = treeMock.patchState.mock.calls[0][0];
    const next = patcher(treeMock._state);
    expect(next.output.plan?.planId).toBe('plan-abc');
    expect(next.output.plan?.passes).toHaveLength(2);
    expect(next.output.plan?.passes[0].passNumber).toBe(1);
    expect(next.output.plan?.passes[0].label).toBe('Form pass');
    expect(next.output.plan?.passes[0].colorHex).toBe('#ff0000');
  });

  it('computePlan pushes plan:compute history', () => {
    service.computePlan();
    expect(treeMock.pushHistory).toHaveBeenCalledWith(expect.stringContaining('plan:compute'));
  });

  it('computePlan passes profile_name when set', () => {
    treeMock.pipeline.mockReturnValue({
      ...treeMock._state.pipeline,
      render: { ...treeMock._state.pipeline.render, profileName: 'brass_60w' },
    });
    service.computePlan();
    expect(apiMock.plan).toHaveBeenCalledWith(
      expect.objectContaining({ profile_name: 'brass_60w' }),
    );
  });
});
