// Pure decisions for keeping per-tab browser state in step with the capture state. The service
// worker applies them; keeping them free of chrome.* makes them unit-testable.
import type { CaptureState } from '../shared/capture-state';

/** The subset of tabs.onUpdated's changeInfo we read. mutedInfo needs no "tabs" permission. */
export interface TabChange {
  mutedInfo?: { muted: boolean };
}

/**
 * The captured tab's new muted state, or undefined when the change isn't about mute or isn't for
 * the tab being captured. Only `capturing` counts: while `starting`, the initial state is read with
 * tabs.get, and a change queued behind the start is applied once it reaches `capturing`.
 */
export function mutedChange(
  state: CaptureState,
  tabId: number,
  change: TabChange,
): boolean | undefined {
  if (state.status !== 'capturing' || state.tabId !== tabId) return undefined;
  return change.mutedInfo?.muted;
}
