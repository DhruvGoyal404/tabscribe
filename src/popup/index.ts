const version = document.getElementById('version');
if (version) {
  version.textContent = `v${chrome.runtime.getManifest().version}`;
}
