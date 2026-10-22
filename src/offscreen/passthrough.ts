/**
 * Time constant for setTargetAtTime when the passthrough mutes or unmutes. The gain is within 2%
 * of its target after about 20 ms (4 time constants): fast enough to feel instant, slow enough
 * not to click.
 */
export const MUTE_RAMP_TIME_CONSTANT = 0.005;

export const passthroughGain = (muted: boolean): number => (muted ? 0 : 1);

/**
 * Ramps the passthrough gain towards the tab's muted state. A new target starts from wherever an
 * earlier ramp has got to, so quick mute/unmute toggles stay smooth.
 */
export function rampPassthrough(
  gain: Pick<AudioParam, 'setTargetAtTime'>,
  muted: boolean,
  now: number,
): void {
  gain.setTargetAtTime(passthroughGain(muted), now, MUTE_RAMP_TIME_CONSTANT);
}
