# Qsarify-Core — backend-only source

Status: public prerelease `v0.1.0-rc.2`, licensed under MIT. Repository: <https://github.com/xHaMMaDy/Qsarify-Core>. This is a backend-only source release; it is not a stable 1.0 API commitment.

## Scope

This repository contains the backend source only under `backend/`: Flask application/API code, services, jobs, backend tests, Python requirements/lockfiles, and the sanitized `.env.example`. It also contains the additive Target Modeling Workspace V2 migrations under `scripts/migrations/`, with operational rollback scripts. It intentionally does **not** contain the Next.js frontend, paper/manuscript, Target Intelligence UI, reviewer demo, production data, ChEMBL caches, user uploads, logs, runtime state, or serialized model artifacts.

The included SQL files are the 12 V2 forward migrations and their matching non-destructive operational rollbacks. They are not a complete Supabase bootstrap or historical schema: a fresh installation still requires QSARify's compatible base schema and Supabase-managed roles/extensions. Review the SQL and execute migrations only against a separate development/staging database until an authorized deployment plan is approved.

## Local setup (Windows)

Use stable Python 3.12; prerelease Python 3.14 is not supported by the pinned scientific wheels.

```powershell
Set-Location backend
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
Copy-Item .env.example .env
```

Fill `.env` only with a dedicated development Supabase project and any feature-specific development provider settings. Never use production credentials in local examples or public issues. Start the local Flask/Waitress entry point with:

```powershell
.\.venv\Scripts\python.exe run.py
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

Two tests that require the separately licensed bundled model binary/metadata skip explicitly when those files are absent. The complete local checkout, where the model fixture exists, passes those two tests as well.

## Release metadata and boundaries

- MIT is the selected license. The maintainer confirmed on 2026-09-24 that the required redistribution clearance is in place for this backend-only source release.
- `CITATION.cff` describes this version. `zenodo.json` is the metadata prepared for DOI archiving; Zenodo assigns the DOI after deposit, so no DOI is claimed yet.
- The MIT license applies to the included software. It does not grant rights to ChEMBL/PubMed data, reviewer labels, manuscript material, QSARify trademarks, or excluded model binaries.
- `CITATION.cff.template` and `zenodo.json.template` are examples for future versions; use the non-template files for `v0.1.0-rc.2`.

The repository and tag are prerelease materials. Verify the exact versioned source and checksum before using it in production or citing it in a manuscript.
