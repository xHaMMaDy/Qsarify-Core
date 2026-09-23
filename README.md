# Qsarify-Core — backend-only repository candidate

Status: local, unpublished staging copy. Intended GitHub destination requested: `xHaMMaDy/Qsarify-Core`. Visibility and license are not yet finalized, so this folder is not pushed.

## Scope

This candidate contains the backend source only under `backend/`: Flask application/API code, services, jobs, backend tests, Python requirements/lockfiles, and the sanitized `.env.example`. It also contains the additive Target Modeling Workspace V2 migrations under `scripts/migrations/`, with operational rollback scripts. It intentionally does **not** contain the Next.js frontend, paper/manuscript, Target Intelligence UI, reviewer demo, repository history, production data, ChEMBL caches, user uploads, logs, runtime state, or serialized model artifacts.

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

## Release decisions still pending

- Confirm whether `Apache-2.0; MIT` means dual license `Apache-2.0 OR MIT`, or choose a single license.
- Confirm all authors/institutional rights holders authorize that license for the backend source.
- Confirm repository visibility (public/private) and the first release version/tag.
- Decide later whether any deployment assets outside the backend should be added; they are intentionally outside this backend-only candidate.
- Citation and Zenodo templates are `CITATION.cff.template` and `zenodo.json.template`; they are not final publication metadata.

No final `LICENSE`, Git remote, Git tag, DOI, or public repository has been created here.
