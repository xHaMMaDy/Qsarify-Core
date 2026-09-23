# Versioning decision for Qsarify-Core

The full-source draft uses `0.1.0-rc.1` as a provisional release-candidate label. SemVer means:

- `0.1.0-rc.1`: first release candidate for version 0.1.0; still pre-release and may change before stable.
- `0.1.0`: first early/stable-compatible package before a 1.0 API stability promise.
- `1.0.0`: first stable release whose documented public interface is intended not to break without a major version bump.
- Increment the patch for backwards-compatible fixes (`0.1.1`), minor for compatible features (`0.2.0`), and major for breaking changes (`1.0.0`). A leading `v` is customary in Git tags (`v0.1.0`), not part of the version itself.

Because `Qsarify-Core` is a newly scoped backend-only repository and its supported migrations/API boundary still need author confirmation, recommendation: use `0.1.0-rc.1` for this review candidate, then `0.1.0` for the first approved early release. The older root `CITATION.cff.template` value `2.0.0` was not treated as a verified release version. Any `v0.1.0-rc.1` tag created in this staging repository is local-only and provisional; do not push it until the source baseline, version, license, and redistribution rights are approved.
