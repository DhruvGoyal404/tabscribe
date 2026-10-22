// Service worker. Phase 2 adds the tabCapture -> offscreen document hand-off here.
chrome.runtime.onInstalled.addListener(({ reason }) => {
  console.info(`[tabscribe] service worker ready (${reason})`);
});
