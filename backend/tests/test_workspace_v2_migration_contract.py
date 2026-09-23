import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "scripts" / "migrations" / "20260913_target_modeling_workspace_v2_foundation.sql"
ROLLBACK = ROOT / "scripts" / "migrations" / "20260913_target_modeling_workspace_v2_foundation.rollback.sql"
STUDIES_MIGRATION = ROOT / "scripts" / "migrations" / "20260913_target_modeling_workspace_v2_studies.sql"
PLAN_MIGRATION = ROOT / "scripts" / "migrations" / "20260913_target_modeling_workspace_v2_plan_review.sql"
COLLECTION_MIGRATION = ROOT / "scripts" / "migrations" / "20260913_target_modeling_workspace_v2_collection_jobs.sql"
TRAINING_REVIEW_MIGRATION = ROOT / "scripts" / "migrations" / "20260913_target_modeling_workspace_v2_training_review.sql"
REGISTRY_MIGRATION = ROOT / "scripts" / "migrations" / "20260913_target_modeling_workspace_v2_registry.sql"
AUDIT_MIGRATION = ROOT / "scripts" / "migrations" / "20260913_target_modeling_workspace_v2_audit.sql"
PUBLIC_QUOTA_MIGRATION = ROOT / "scripts" / "migrations" / "20260913_target_modeling_workspace_v2_public_quota.sql"
UPLOADED_MODELS_MIGRATION = ROOT / "scripts" / "migrations" / "20260914_target_modeling_workspace_v2_uploaded_models.sql"
ACCOUNT_RETENTION_MIGRATION = ROOT / "scripts" / "migrations" / "20260914_target_modeling_workspace_v2_account_retention.sql"
QUOTA_DEFAULTS_MIGRATION = ROOT / "scripts" / "migrations" / "20260914_target_modeling_workspace_v2_quota_defaults.sql"
CHAT_QUOTA_MIGRATION = ROOT / "scripts" / "migrations" / "20260914_target_modeling_workspace_v2_chat_quota.sql"


def test_workspace_v2_migration_is_additive_disabled_and_owner_scoped():
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    assert "target_modeling_workspace_v2 boolean not null default false" in sql
    assert "create table if not exists public.ti_workspace_members" in sql
    assert "create table if not exists public.ti_workspace_target_sources" in sql
    assert "create table if not exists public.ti_job_events" in sql
    assert "alter table public.ti_job_events enable row level security" in sql
    assert "public.can_access_ti_workspace" in sql
    assert "create_ti_workspace_owner_membership" in sql
    assert "grant select on public.ti_job_events to authenticated" in sql
    assert "revoke all on public.ti_job_events from anon" in sql
    assert "alter publication supabase_realtime add table public.ti_job_events" in sql
    assert "service_role" not in sql


def test_workspace_v2_operational_rollback_preserves_data():
    sql = ROLLBACK.read_text(encoding="utf-8").lower()

    assert "set target_modeling_workspace_v2 = false" in sql
    assert re.search(r"(?m)^\s*drop table\s", sql) is None
    assert "drop column" not in sql
    assert "alter publication supabase_realtime drop table public.ti_job_events" in sql


def test_workspace_v2_studies_migration_supports_members_without_mutable_snapshots():
    sql = STUDIES_MIGRATION.read_text(encoding="utf-8").lower()

    assert "ti members read studies" in sql
    assert "ti editors update studies" in sql
    assert "ti owners delete studies" in sql
    assert "ti members read study targets" in sql
    assert "ti editors add study targets" in sql
    assert "ti editors remove study targets" in sql
    assert "for update to authenticated" not in sql.split('on public.ti_workspace_targets', 1)[1]
    assert "report.run_id = run_id" in sql
    assert "workspace.user_id = user_id" in sql
    assert "ti members read study datasets" in sql
    assert "ti members read study training runs" in sql
    assert "ti members read study artifacts" in sql


def test_workspace_v2_plan_migration_freezes_review_metadata_and_member_access():
    sql = PLAN_MIGRATION.read_text(encoding="utf-8").lower()

    assert "confirmed_version_id" in sql
    assert "execution_snapshot jsonb" in sql
    assert "input_dossier_hash" in sql
    assert "recommendation_hash" in sql
    assert "deterministic_validation jsonb" in sql
    assert "ti editors update study plans" in sql
    assert "ti editors create study plan versions" in sql


def test_workspace_v2_collection_migration_is_per_target_and_versioned():
    sql = COLLECTION_MIGRATION.read_text(encoding="utf-8").lower()

    assert "create table if not exists public.ti_collection_jobs" in sql
    assert "workspace_target_id uuid not null" in sql
    assert "idempotency_key text not null" in sql
    assert "'retrying'" in sql
    assert "content_sha256" in sql
    assert "immutable_at" in sql
    assert "ti editors create collection jobs" in sql
    assert "ti members read collection jobs" in sql
    assert "grant select, insert, update on public.ti_collection_jobs to authenticated" in sql


def test_workspace_v2_training_review_migration_supports_both_and_candidate_gate():
    sql = TRAINING_REVIEW_MIGRATION.read_text(encoding="utf-8").lower()

    assert "parent_run_id uuid" in sql
    assert "execution_scope text" in sql
    assert "'separate_child'" in sql
    assert "'pooled_child'" in sql
    assert "architecture in ('separate_models', 'pooled_multitarget', 'both')" in sql
    assert "ready_for_review" in sql
    assert "ti reviewers update study artifacts" in sql


def test_workspace_v2_registry_migration_freezes_public_and_artifact_boundaries():
    sql = REGISTRY_MIGRATION.read_text(encoding="utf-8").lower()

    assert "checksum_sha256" in sql
    assert "manifest jsonb" in sql
    assert "storage_key" in sql
    assert "public_slug" in sql
    assert "visibility in ('unlisted', 'public')" in sql
    assert "allow_public_download boolean not null default false" in sql
    assert "public_prediction_rate" not in sql
    assert "public_prediction" in sql
    assert "chat_request" in sql
    assert "revoke all on public.ti_model_artifacts from anon" in sql


def test_workspace_v2_audit_migration_scopes_events_to_studies():
    sql = AUDIT_MIGRATION.read_text(encoding="utf-8").lower()

    assert "workspace_id uuid references public.ti_workspaces" in sql
    assert "ti members read study audit history" in sql
    assert "ti members write study audit history" in sql
    assert "actor_user_id = (select auth.uid())" in sql
    assert "grant select, insert on public.ti_audit_events to authenticated" in sql


def test_workspace_v2_public_quota_is_atomic_and_service_only():
    sql = PUBLIC_QUOTA_MIGRATION.read_text(encoding="utf-8").lower()

    assert "create table if not exists public.ti_public_prediction_buckets" in sql
    assert "consume_ti_public_prediction_quota" in sql
    assert "pg_advisory_xact_lock" in sql
    assert "grant execute on function public.consume_ti_public_prediction_quota" in sql
    assert "to service_role" in sql
    assert "revoke all on public.ti_public_prediction_buckets from public, anon, authenticated" in sql


def test_workspace_v2_uploaded_models_are_owner_scoped_and_confirmation_gated():
    sql = UPLOADED_MODELS_MIGRATION.read_text(encoding="utf-8").lower()

    assert "create table if not exists public.ti_uploaded_models" in sql
    assert "storage_key text not null unique" in sql
    assert "metadata_confirmed boolean not null default false" in sql
    assert "user_id = (select auth.uid())" in sql
    assert "revoke all on public.ti_uploaded_models from anon" in sql


def test_workspace_v2_account_retention_is_fixed_owner_scoped_and_non_destructive():
    sql = ACCOUNT_RETENTION_MIGRATION.read_text(encoding="utf-8").lower()

    assert "create table if not exists public.ti_account_deletion_requests" in sql
    assert "interval '180 days'" in sql
    assert "status in ('scheduled', 'cancelled', 'executed', 'failed')" in sql
    assert "user_id = (select auth.uid())" in sql
    assert "revoke all on public.ti_account_deletion_requests from anon" in sql
    assert "drop table" not in sql


def test_workspace_v2_quota_defaults_preserve_existing_admin_values():
    sql = QUOTA_DEFAULTS_MIGRATION.read_text(encoding="utf-8").lower()

    assert "alter column public_daily_analyses set default 10" in sql
    assert "update public.ti_admin_settings" not in sql
    assert "existing administrator values are preserved" in sql


def test_chat_quota_capabilities_preserve_workspace_v2_flag():
    sql = CHAT_QUOTA_MIGRATION.read_text(encoding="utf-8").lower()
    assert "target_modeling_workspace_v2 boolean" in sql
    assert "s.target_modeling_workspace_v2" in sql
    assert "public_daily_chat_requests" in sql


def test_every_workspace_v2_forward_migration_has_a_data_preserving_rollback():
    migration_dir = ROOT / "scripts" / "migrations"
    forwards = sorted(path for path in migration_dir.glob("*target_modeling_workspace_v2_*.sql") if not path.name.endswith(".rollback.sql"))
    assert forwards
    for forward in forwards:
        rollback = forward.with_name(forward.name.removesuffix(".sql") + ".rollback.sql")
        assert rollback.is_file(), f"Missing rollback for {forward.name}"
        rollback_sql = rollback.read_text(encoding="utf-8").lower()
        assert any(marker in rollback_sql for marker in ("set target_modeling_workspace_v2 = false", "revoke", "drop policy", "allow_public_predictions = false", "set default 3", "public_daily_chat_requests = 0")), f"Rollback for {forward.name} has no operational disable action"
        assert re.search(r"(?m)^\s*drop\s+table\s", rollback_sql) is None
        assert re.search(r"(?m)^\s*drop\s+column\s", rollback_sql) is None
