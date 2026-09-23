-- Target Modeling Workspace V2 P3: durable per-target collection jobs and
-- immutable dataset metadata.
-- Requires the P0/P1/P2 Workspace V2 migrations.
-- Operational rollback: disable V2 and stop collection workers. Existing
-- datasets are preserved; no destructive schema rollback is automatic.

BEGIN;

CREATE TABLE IF NOT EXISTS public.ti_collection_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  workspace_id UUID NOT NULL REFERENCES public.ti_workspaces(id) ON DELETE CASCADE,
  workspace_target_id UUID NOT NULL REFERENCES public.ti_workspace_targets(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN (
    'queued', 'validating', 'collecting', 'completed', 'failed',
    'cancelling', 'cancelled', 'retrying'
  )),
  requested_config JSONB NOT NULL DEFAULT '{}'::jsonb,
  progress_percent INTEGER NOT NULL DEFAULT 0 CHECK (progress_percent BETWEEN 0 AND 100),
  progress_stage TEXT NOT NULL DEFAULT 'queued',
  record_count INTEGER NOT NULL DEFAULT 0 CHECK (record_count >= 0),
  error_code TEXT,
  error_message TEXT CHECK (error_message IS NULL OR char_length(error_message) <= 500),
  idempotency_key TEXT NOT NULL CHECK (char_length(idempotency_key) BETWEEN 8 AND 200),
  attempt_count INTEGER NOT NULL DEFAULT 1 CHECK (attempt_count >= 1),
  max_attempts INTEGER NOT NULL DEFAULT 3 CHECK (max_attempts BETWEEN 1 AND 5),
  cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
  queued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  started_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (workspace_target_id, idempotency_key)
);

ALTER TABLE public.ti_training_datasets
  ADD COLUMN IF NOT EXISTS content_sha256 TEXT,
  ADD COLUMN IF NOT EXISTS parent_dataset_id UUID REFERENCES public.ti_training_datasets(id) ON DELETE SET NULL,
  ADD COLUMN IF NOT EXISTS source_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS endpoint_groups JSONB NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS feature_schema JSONB NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS immutable_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS superseded_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS curation_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS preparation_summary JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE public.ti_training_datasets DROP CONSTRAINT IF EXISTS ti_training_datasets_status_check;
ALTER TABLE public.ti_training_datasets ADD CONSTRAINT ti_training_datasets_status_check CHECK (status IN (
  'planned', 'collecting', 'collected', 'curating', 'curated', 'preparing',
  'ready_for_training', 'failed', 'expired', 'superseded'
));

CREATE INDEX IF NOT EXISTS idx_ti_collection_jobs_workspace
  ON public.ti_collection_jobs(workspace_id, queued_at ASC);
CREATE INDEX IF NOT EXISTS idx_ti_collection_jobs_queue
  ON public.ti_collection_jobs(status, queued_at ASC);
CREATE INDEX IF NOT EXISTS idx_ti_collection_jobs_target
  ON public.ti_collection_jobs(workspace_target_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ti_training_datasets_hash
  ON public.ti_training_datasets(content_sha256)
  WHERE content_sha256 IS NOT NULL;

DROP TRIGGER IF EXISTS ti_collection_jobs_updated_at ON public.ti_collection_jobs;
CREATE TRIGGER ti_collection_jobs_updated_at
  BEFORE UPDATE ON public.ti_collection_jobs
  FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

ALTER TABLE public.ti_collection_jobs ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "TI members read collection jobs" ON public.ti_collection_jobs;
CREATE POLICY "TI members read collection jobs"
  ON public.ti_collection_jobs FOR SELECT TO authenticated
  USING (public.can_access_ti_workspace(workspace_id, NULL));

DROP POLICY IF EXISTS "TI editors create collection jobs" ON public.ti_collection_jobs;
CREATE POLICY "TI editors create collection jobs"
  ON public.ti_collection_jobs FOR INSERT TO authenticated
  WITH CHECK (
    user_id = (SELECT auth.uid())
    AND public.can_access_ti_workspace(workspace_id, ARRAY['owner', 'admin', 'editor'])
    AND EXISTS (
      SELECT 1 FROM public.ti_workspace_targets AS target
      WHERE target.id = workspace_target_id
        AND target.workspace_id = workspace_id
    )
  );

DROP POLICY IF EXISTS "TI editors update collection jobs" ON public.ti_collection_jobs;
CREATE POLICY "TI editors update collection jobs"
  ON public.ti_collection_jobs FOR UPDATE TO authenticated
  USING (public.can_access_ti_workspace(workspace_id, ARRAY['owner', 'admin', 'editor']))
  WITH CHECK (public.can_access_ti_workspace(workspace_id, ARRAY['owner', 'admin', 'editor']));

GRANT SELECT, INSERT, UPDATE ON public.ti_collection_jobs TO authenticated;
GRANT SELECT ON public.ti_training_datasets TO authenticated;

DROP POLICY IF EXISTS "TI members read Study datasets" ON public.ti_training_datasets;
CREATE POLICY "TI members read Study datasets"
  ON public.ti_training_datasets FOR SELECT TO authenticated
  USING (public.can_access_ti_workspace(workspace_id, NULL));

COMMENT ON TABLE public.ti_collection_jobs IS
  'Durable, retryable, per-target ChEMBL collection jobs inside a Study.';
COMMENT ON COLUMN public.ti_training_datasets.content_sha256 IS
  'Canonical hash of the persisted raw dataset snapshot.';

COMMIT;
