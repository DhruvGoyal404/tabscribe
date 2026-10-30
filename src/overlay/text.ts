// What the overlay shows: the newest final and the partial that follows it, as the popup does.
import type { Caption, CaptionStatus } from '../shared/captions';

export interface OverlayText {
  final: string;
  partial: string;
}

export const EMPTY: OverlayText = { final: '', partial: '' };

export function applyCaption(text: OverlayText, caption: Caption): OverlayText {
  // An empty final (a dropped decode) keeps the last shown text but ends the partial it replaces.
  if (caption.kind === 'final') return { final: caption.text || text.final, partial: '' };
  return { ...text, partial: caption.text };
}

/** Text from before a restart or a failure is stale: only a listening pipeline keeps it. */
export const applyStatus = (text: OverlayText, status: CaptionStatus): OverlayText =>
  status.state === 'listening' ? text : EMPTY;
