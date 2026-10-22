import { describe, expect, it, vi } from 'vitest';
import { MUTE_RAMP_TIME_CONSTANT, passthroughGain, rampPassthrough } from './passthrough';

describe('passthroughGain', () => {
  it('is silent when muted and unity otherwise', () => {
    expect(passthroughGain(true)).toBe(0);
    expect(passthroughGain(false)).toBe(1);
  });
});

describe('rampPassthrough', () => {
  it.each([
    [true, 0],
    [false, 1],
  ])('muted=%s ramps to %s from the current time', (muted, target) => {
    const gain = { setTargetAtTime: vi.fn() };
    rampPassthrough(gain, muted, 12.5);
    expect(gain.setTargetAtTime).toHaveBeenCalledExactlyOnceWith(
      target,
      12.5,
      MUTE_RAMP_TIME_CONSTANT,
    );
  });

  it('settles within about 20 ms', () => {
    // setTargetAtTime approaches the target as 1 - e^(-t/τ).
    expect(1 - Math.exp(-0.02 / MUTE_RAMP_TIME_CONSTANT)).toBeGreaterThan(0.98);
  });
});
