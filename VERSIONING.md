# Versioning decision for Qsarify-Core

The backend-only public prerelease is `0.1.0-rc.2`. SemVer means:

- `0.1.0-rc.1`: the initial private review candidate; its tag is retained unchanged.
- `0.1.0-rc.2`: the second release candidate, containing the same backend implementation with finalized public-release metadata and rights-clearance record; still pre-release and may change before stable.
- `0.1.0`: first early/stable-compatible package before a 1.0 API stability promise.
- `1.0.0`: first stable release whose documented public interface is intended not to break without a major version bump.
- Increment the patch for backwards-compatible fixes (`0.1.1`), minor for compatible features (`0.2.0`), and major for breaking changes (`1.0.0`). A leading `v` is customary in Git tags (`v0.1.0`), not part of the version itself.

The older root `CITATION.cff.template` value `2.0.0` was not treated as a verified release version. `v0.1.0-rc.2` is the approved public prerelease tag for this backend-only scope. Use `0.1.0` for a future stable-compatible release after validation and reserve `1.0.0` for a stable API commitment.
