-- Target Modeling Workspace V2 P1: multiple Studies, collaboration access,
-- immutable target snapshots, and archive lifecycle.
-- Requires 20260913_target_modeling_workspace_v2_foundation.sql.
-- Operational rollback: disable target_modeling_workspace_v2. A destructive
-- schema rollback is intentionally deferred until the retention window ends.

BEGIN;

ALTER TABLE public.ti_workspace_targets
  ADD COLUMN IF NOT EXISTS added_by UUID REFERENCES auth.users(id) ON DELETE SET NULL;

UPDATE public.ti_workspace_targets
SET added_by = user_id
WHERE added_by IS NULL;

ALTER TABLE public.ti_workspaces
  DROP CONSTRAINT IF EXISTS ti_workspaces_archive_fields_check;
ALTER TABLE public.ti_workspaces
  ADD CONSTRAINT ti_workspaces_archive_fields_check CHECK (
    status <> 'archived'
    OR (archived_at IS NOT NULL AND archive_expires_at IS NOT NULL)
  );

DROP POLICY IF EXISTS "TI users manage own workspaces" ON public.ti_workspaces;
DROP POLICY IF EXISTS "TI members read Studies" ON public.ti_workspaces;
CREATE POLICY "TI members read Studies"
  ON public.ti_workspaces FOR SELECT TO authenticated
  USING (public.can_access_ti_workspace(id, NULL));

DROP POLICY IF EXISTS "TI users create own Studies" ON public.ti_workspaces;
CREATE POLICY "TI users create own Studies"
  ON public.ti_workspaces FOR INSERT TO authenticated
  WITH CHECK (user_id = (SELECT auth.uid()));

DROP POLICY IF EXISTS "TI editors update Studies" ON public.ti_workspaces;
CREATE POLICY "TI editors update Studies"
  ON public.ti_workspaces FOR UPDATE TO authenticated
  USING (public.can_access_ti_workspace(id, ARRAY['owner', 'admin', 'editor']))
  WITH CHECK (public.can_access_ti_workspace(id, ARRAY['owner', 'admin', 'editor']));

DROP POLICY IF EXISTS "TI owners delete Studies" ON public.ti_workspaces;
CREATE POLICY "TI owners delete Studies"
  ON public.ti_workspaces FOR DELETE TO authenticated
  USING (user_id = (SELECT auth.uid()) OR COALESCE(public.is_admin(), FALSE));

DROP POLICY IF EXISTS "TI users manage own workspace targets" ON public.ti_workspace_targets;
DROP POLICY IF EXISTS "TI members read Study targets" ON public.ti_workspace_targets;
CREATE POLICY "TI members read Study targets"
  ON public.ti_workspace_targets FOR SELECT TO authenticated
  USING (public.can_access_ti_workspace(workspace_id, NULL));

DROP POLICY IF EXISTS "TI editors add Study targets" ON public.ti_workspace_targets;
CREATE POLICY "TI editors add Study targets"
  ON public.ti_workspace_targets FOR INSERT TO authenticated
  WITH CHECK (
    added_by = (SELECT auth.uid())
    AND public.can_access_ti_workspace(workspace_id, ARRAY['owner', 'admin', 'editor'])
    AND EXISTS (
      SELECT 1 FROM public.ti_workspaces AS workspace
      WHERE workspace.id = workspace_id
        AND workspace.user_id = user_id
        AND workspace.deleted_at IS NULL
    )
    AND EXISTS (
      SELECT 1 FROM public.ti_reports AS report
      WHERE report.id = report_id
        AND report.user_id = (SELECT auth.uid())
    )
  );

DROP POLICY IF EXISTS "TI editors remove Study targets" ON public.ti_workspace_targets;
CREATE POLICY "TI editors remove Study targets"
  ON public.ti_workspace_targets FOR DELETE TO authenticated
  USING (public.can_access_ti_workspace(workspace_id, ARRAY['owner', 'admin', 'editor']));

DROP POLICY IF EXISTS "TI editors add target provenance" ON public.ti_workspace_target_sources;
CREATE POLICY "TI editors add target provenance"
  ON public.ti_workspace_target_sources FOR INSERT TO authenticated
  WITH CHECK (
    added_by = (SELECT auth.uid())
    AND public.can_access_ti_workspace(workspace_id, ARRAY['owner', 'admin', 'editor'])
    AND EXISTS (
      SELECT 1 FROM public.ti_reports AS report
      WHERE report.id = report_id
        AND report.run_id = run_id
        AND report.user_id = (SELECT auth.uid())
    )
  );

CREATE INDEX IF NOT EXISTS idx_ti_workspaces_archive_expiry
  ON public.ti_workspaces(archive_expires_at)
  WHERE status = 'archived' AND deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_ti_workspace_targets_added_by
  ON public.ti_workspace_targets(added_by, created_at DESC);

REVOKE ALL ON public.ti_workspaces FROM anon;
REVOKE ALL ON public.ti_workspace_targets FROM anon;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.ti_workspaces TO authenticated;
GRANT SELECT, INSERT, DELETE ON public.ti_workspace_targets TO authenticated;
GRANT SELECT ON public.ti_runs, public.ti_reports TO authenticated;

-- Existing workspace entities retain owner policies from 20260912. Add member
-- read access for the Study shell while preserving role-gated mutations for
-- later P2-P5 execution migrations.
GRANT SELECT ON public.ti_study_plans, public.ti_study_plan_versions,
  public.ti_training_datasets, public.ti_training_runs,
  public.ti_model_artifacts, public.ti_model_deployments TO authenticated;

DROP POLICY IF EXISTS "TI members read Study plans" ON public.ti_study_plans;
CREATE POLICY "TI members read Study plans"
  ON public.ti_study_plans FOR SELECT TO authenticated
  USING (public.can_access_ti_workspace(workspace_id, NULL));

DROP POLICY IF EXISTS "TI members read Study plan versions" ON public.ti_study_plan_versions;
CREATE POLICY "TI members read Study plan versions"
  ON public.ti_study_plan_versions FOR SELECT TO authenticated
  USING (EXISTS (
    SELECT 1 FROM public.ti_study_plans AS plan
    WHERE plan.id = study_plan_id
      AND public.can_access_ti_workspace(plan.workspace_id, NULL)
  ));

DROP POLICY IF EXISTS "TI members read Study datasets" ON public.ti_training_datasets;
CREATE POLICY "TI members read Study datasets"
  ON public.ti_training_datasets FOR SELECT TO authenticated
  USING (public.can_access_ti_workspace(workspace_id, NULL));

DROP POLICY IF EXISTS "TI members read Study training runs" ON public.ti_training_runs;
CREATE POLICY "TI members read Study training runs"
  ON public.ti_training_runs FOR SELECT TO authenticated
  USING (public.can_access_ti_workspace(workspace_id, NULL));

DROP POLICY IF EXISTS "TI members read Study artifacts" ON public.ti_model_artifacts;
CREATE POLICY "TI members read Study artifacts"
  ON public.ti_model_artifacts FOR SELECT TO authenticated
  USING (public.can_access_ti_workspace(workspace_id, NULL));

DROP POLICY IF EXISTS "TI members read Study deployments" ON public.ti_model_deployments;
CREATE POLICY "TI members read Study deployments"
  ON public.ti_model_deployments FOR SELECT TO authenticated
  USING (public.can_access_ti_workspace(workspace_id, NULL));

COMMENT ON COLUMN public.ti_workspace_targets.added_by IS
  'Actor who selected the immutable report-backed target snapshot.';

COMMIT;
