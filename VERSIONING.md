# Versioning decision for Qsarify-Core

The full-source draft uses `0.1.0-rc.1` as a provisional release-candidate label. SemVer means:

- `0.1.0-rc.1`: first release candidate for version 0.1.0; still pre-release and may change before stable.
- `0.1.0`: first early/stable-compatible package before a 1.0 API stability promise.
- `1.0.0`: first stable release whose documented public interface is intended not to break without a major version bump.
- Increment the patch for backwards-compatible fixes (`0.1.1`), minor for compatible features (`0.2.0`), and major for breaking changes (`1.0.0`). A leading `v` is customary in Git tags (`v0.1.0`), not part of the version itself.

Because `Qsarify-Core` is a newly scoped backend-only repository and its supported migrations/API boundary still need author confirmation, the user selected `0.1.0-rc.1` for the review candidate. The older root `CITATION.cff.template` value `2.0.0` was not treated as a verified release version. The tag may be pushed to the private staging repository after local/production source synchronization; keep the repository private until author/institutional redistribution rights and public-demo gates are approved. Use `0.1.0` for the first approved early release, and reserve `1.0.0` for a stable API commitment.
