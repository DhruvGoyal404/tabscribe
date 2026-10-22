// Pure decisions for keeping per-tab browser state in step with the capture state. The service
// worker applies them; keeping them free of chrome.* makes them unit-testable.
import type { CaptureState } from '../shared/capture-state';

/** The subset of tabs.onUpdated's changeInfo we read. Neither field needs the "tabs" permission. */
export interface TabChange {
  mutedInfo?: { muted: boolean };
  status?: string;
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

export const BADGE_TEXT = 'ON';

export interface BadgeUpdate {
  tabId: number;
  text: string;
}

/** The tab that should show the badge: only a tab that is actually being captured. */
export const badgeTabId = (state: CaptureState): number | undefined =>
  state.status === 'capturing' ? state.tabId : undefined;

/**
 * Per-tab badge changes for a state transition: clear the tab that stops showing it, then set the
 * one that starts. Per-tab, so a second window's action icon never claims its own tab is captured.
 */
export function badgeUpdates(previous: CaptureState, next: CaptureState): BadgeUpdate[] {
  const before = badgeTabId(previous);
  const after = badgeTabId(next);
  if (before === after) return [];
  const updates: BadgeUpdate[] = [];
  if (before !== undefined) updates.push({ tabId: before, text: '' });
  if (after !== undefined) updates.push({ tabId: after, text: BADGE_TEXT });
  return updates;
}

/**
 * Chrome clears per-tab badge text when the tab navigates (verified empirically; the docs only
 * mention tab close), but capture survives navigation. Re-apply the badge on the captured tab's
 * loading-status changes: both `loading` and `complete`, so it holds whichever one Chrome clears at.
 */
export function badgeRefresh(
  state: CaptureState,
  tabId: number,
  change: TabChange,
): BadgeUpdate | undefined {
  return change.status !== undefined && badgeTabId(state) === tabId
    ? { tabId, text: BADGE_TEXT }
    : undefined;
}
