-- Target Modeling Workspace V2 P4: training scope, parent/child runs, and
-- explicit candidate review states.
-- Requires P0-P3 Workspace V2 migrations.
-- Operational rollback: stop new V2 runs and disable the feature flag. Existing
-- run and artifact history remains preserved.

BEGIN;

ALTER TABLE public.ti_training_runs
  ADD COLUMN IF NOT EXISTS parent_run_id UUID REFERENCES public.ti_training_runs(id) ON DELETE SET NULL,
  ADD COLUMN IF NOT EXISTS execution_scope TEXT NOT NULL DEFAULT 'single',
  ADD COLUMN IF NOT EXISTS confirmed_config_hash TEXT,
  ADD COLUMN IF NOT EXISTS configuration_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS event_sequence BIGINT NOT NULL DEFAULT 0;

ALTER TABLE public.ti_training_runs DROP CONSTRAINT IF EXISTS ti_training_runs_architecture_check;
ALTER TABLE public.ti_training_runs ADD CONSTRAINT ti_training_runs_architecture_check CHECK (
  architecture IN ('separate_models', 'pooled_multitarget', 'both')
);

ALTER TABLE public.ti_training_runs DROP CONSTRAINT IF EXISTS ti_training_runs_execution_scope_check;
ALTER TABLE public.ti_training_runs ADD CONSTRAINT ti_training_runs_execution_scope_check CHECK (
  execution_scope IN ('single', 'parent', 'separate_child', 'pooled_child')
);

ALTER TABLE public.ti_training_runs DROP CONSTRAINT IF EXISTS ti_training_runs_status_check;
ALTER TABLE public.ti_training_runs ADD CONSTRAINT ti_training_runs_status_check CHECK (status IN (
  'queued', 'validating', 'collecting', 'curating', 'featurizing', 'training',
  'evaluating', 'completed', 'ready_for_review', 'failed', 'retrying',
  'cancelling', 'cancelled', 'expired'
));

ALTER TABLE public.ti_model_artifacts
  ADD COLUMN IF NOT EXISTS approved_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS approved_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  ADD COLUMN IF NOT EXISTS disabled_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS training_config_hash TEXT;

ALTER TABLE public.ti_model_artifacts DROP CONSTRAINT IF EXISTS ti_model_artifacts_architecture_check;
ALTER TABLE public.ti_model_artifacts ADD CONSTRAINT ti_model_artifacts_architecture_check CHECK (
  architecture IN ('separate_models', 'pooled_multitarget', 'both')
);

ALTER TABLE public.ti_model_artifacts DROP CONSTRAINT IF EXISTS ti_model_artifacts_status_check;
ALTER TABLE public.ti_model_artifacts ADD CONSTRAINT ti_model_artifacts_status_check CHECK (status IN (
  'trained', 'training', 'ready_for_review', 'approved', 'rejected',
  'deploying', 'deployed', 'disabled', 'deactivated', 'failed'
));

ALTER TABLE public.ti_model_deployments DROP CONSTRAINT IF EXISTS ti_model_deployments_architecture_check;
ALTER TABLE public.ti_model_deployments ADD CONSTRAINT ti_model_deployments_architecture_check CHECK (
  architecture IN ('separate_models', 'pooled_multitarget', 'both')
);

CREATE INDEX IF NOT EXISTS idx_ti_training_runs_parent
  ON public.ti_training_runs(parent_run_id, created_at DESC)
  WHERE parent_run_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ti_training_runs_scope
  ON public.ti_training_runs(workspace_id, execution_scope, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_ti_model_artifacts_review
  ON public.ti_model_artifacts(workspace_id, status, created_at DESC);

DROP POLICY IF EXISTS "TI members read Study training runs" ON public.ti_training_runs;
CREATE POLICY "TI members read Study training runs"
  ON public.ti_training_runs FOR SELECT TO authenticated
  USING (public.can_access_ti_workspace(workspace_id, NULL));

DROP POLICY IF EXISTS "TI editors create Study training runs" ON public.ti_training_runs;
CREATE POLICY "TI editors create Study training runs"
  ON public.ti_training_runs FOR INSERT TO authenticated
  WITH CHECK (
    user_id = (SELECT auth.uid())
    AND public.can_access_ti_workspace(workspace_id, ARRAY['owner', 'admin', 'editor'])
  );

DROP POLICY IF EXISTS "TI editors update Study training runs" ON public.ti_training_runs;
CREATE POLICY "TI editors update Study training runs"
  ON public.ti_training_runs FOR UPDATE TO authenticated
  USING (public.can_access_ti_workspace(workspace_id, ARRAY['owner', 'admin', 'editor']))
  WITH CHECK (public.can_access_ti_workspace(workspace_id, ARRAY['owner', 'admin', 'editor']));

DROP POLICY IF EXISTS "TI members read Study artifacts" ON public.ti_model_artifacts;
CREATE POLICY "TI members read Study artifacts"
  ON public.ti_model_artifacts FOR SELECT TO authenticated
  USING (public.can_access_ti_workspace(workspace_id, NULL));

DROP POLICY IF EXISTS "TI reviewers update Study artifacts" ON public.ti_model_artifacts;
CREATE POLICY "TI reviewers update Study artifacts"
  ON public.ti_model_artifacts FOR UPDATE TO authenticated
  USING (public.can_access_ti_workspace(workspace_id, ARRAY['owner', 'admin', 'reviewer']))
  WITH CHECK (public.can_access_ti_workspace(workspace_id, ARRAY['owner', 'admin', 'reviewer']));

GRANT SELECT, INSERT, UPDATE ON public.ti_training_runs TO authenticated;
GRANT SELECT, UPDATE ON public.ti_model_artifacts TO authenticated;

COMMENT ON COLUMN public.ti_training_runs.execution_scope IS
  'single, parent, separate_child, or pooled_child for explicit both-architecture execution.';
COMMENT ON COLUMN public.ti_model_artifacts.status IS
  'Candidate lifecycle; ready_for_review is the post-training review gate.';

COMMIT;

