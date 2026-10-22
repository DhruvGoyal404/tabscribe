# Security policy

## Supported versions

Tabscribe is pre-release. Only the latest commit on `main` gets security fixes.

## Reporting a vulnerability

Please **don't** open a public issue. Report it privately through GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on this repository (Security tab → "Report a vulnerability").

Include the affected version or commit, steps to reproduce, and the impact. You should get an
acknowledgement within 7 days.

## Scope and design principles

- Least privilege: the manifest requests only `activeTab`, `tabCapture`, `offscreen`, `scripting`
  and `storage`, and no host permissions.
- No remote code: all scripts and WebAssembly are bundled. Extension pages use the CSP
  `script-src 'self' 'wasm-unsafe-eval'; object-src 'self'`.
- No secrets in the repository. gitleaks scans every push.
- Dependencies are pinned exactly, must be at least 14 days old, and must pass `npm audit` at
  high severity.
