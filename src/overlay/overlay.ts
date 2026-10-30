// Caption overlay in the captured tab. The service worker injects it on demand (chrome.scripting,
// under the activeTab grant from the popup click), into the extension's isolated world.
//
// - A closed shadow root keeps the page's CSS and scripts away from it; text goes in through
//   textContent only, never as markup.
// - Styles go through a constructed stylesheet and the CSSOM, which a page's CSP (style-src)
//   doesn't govern, unlike a <style> element or a style attribute in markup.
// - It shows what the offscreen document sends over the overlay port, and removes itself when
//   that port closes: Stop, the end of the capture, or the extension reloading.
import { isCaptionMessage } from '../shared/captions';
import { OVERLAY_PORT } from '../shared/messages';
import { applyCaption, applyStatus, EMPTY, type OverlayText } from './text';

// Sizes are in px and viewport units, not rem, which would follow the page's root font size.
// White on 0.8 black stays above 12:1 even on a white page, and a partial above 7:1.
const CSS = `
  .captions {
    position: fixed;
    inset-inline: 0;
    bottom: max(8vh, 72px); /* clear of a player's controls */
    width: fit-content;
    max-width: min(90vw, 34em);
    margin-inline: auto;
    padding: 0.4em 0.7em;
    border-radius: 0.3em;
    background: rgb(0 0 0 / 0.8);
    color: #fff;
    font: 500 clamp(18px, 2.2vw, 36px) / 1.35 system-ui, sans-serif;
    text-align: center;
    pointer-events: none;
  }
  .captions[hidden] {
    display: none;
  }
  /* Three lines at most, anchored at the end: the oldest text goes first. */
  .lines {
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    max-height: 3lh;
    overflow: hidden;
  }
  .partial {
    opacity: 0.7;
  }
`;

/** Survives in the isolated world between injections, so a second one replaces the first. */
const INSTANCE = '__tabscribeOverlay';
type Instances = Record<typeof INSTANCE, { remove(): void } | undefined>;
const global = globalThis as unknown as Instances;

function mount(): void {
  global[INSTANCE]?.remove();

  const host = document.createElement('tabscribe-overlay');
  // Inline !important beats any page rule, so the page can't hide or move the host.
  host.style.setProperty('all', 'initial', 'important');
  host.style.setProperty('position', 'fixed', 'important');
  host.style.setProperty('inset', '0', 'important');
  host.style.setProperty('z-index', '2147483647', 'important');
  host.style.setProperty('pointer-events', 'none', 'important');

  const root = host.attachShadow({ mode: 'closed' });
  const sheet = new CSSStyleSheet();
  sheet.replaceSync(CSS);
  root.adoptedStyleSheets = [sheet];
  const box = document.createElement('div');
  box.className = 'captions';
  box.hidden = true;
  const final = document.createElement('span');
  const partial = document.createElement('span');
  partial.className = 'partial';
  const lines = document.createElement('div');
  lines.className = 'lines';
  const line = document.createElement('div');
  line.className = 'text';
  line.append(final, ' ', partial);
  lines.append(line);
  box.append(lines);
  root.append(box);
  document.documentElement.append(host);

  let text: OverlayText = EMPTY;
  const show = () => {
    final.textContent = text.final;
    partial.textContent = text.partial;
    box.hidden = !text.final && !text.partial;
  };

  const port = chrome.runtime.connect({ name: OVERLAY_PORT });
  const instance = {
    remove() {
      port.disconnect();
      host.remove();
      if (global[INSTANCE] === instance) global[INSTANCE] = undefined;
    },
  };
  global[INSTANCE] = instance;
  port.onMessage.addListener((message: unknown) => {
    if (!isCaptionMessage(message)) return;
    text =
      message.type === 'captions/caption'
        ? applyCaption(text, message.caption)
        : applyStatus(text, message.status);
    show();
  });
  port.onDisconnect.addListener(() => {
    // Reading lastError marks it handled (the offscreen document may never have existed).
    if (chrome.runtime.lastError) console.debug('[tabscribe] overlay port closed');
    instance.remove();
  });
}

mount();
