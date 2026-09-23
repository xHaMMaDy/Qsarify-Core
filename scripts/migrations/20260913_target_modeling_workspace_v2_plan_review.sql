-- Target Modeling Workspace V2 P2: auditable AI plan review metadata.
-- Requires the P0 foundation and P1 Studies migration.
-- Rollback: disable V2 and preserve plan versions; no plan data is deleted.

BEGIN;

ALTER TABLE public.ti_study_plans
  ADD COLUMN IF NOT EXISTS confirmed_version_id UUID,
  ADD COLUMN IF NOT EXISTS confirmed_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS confirmed_by UUID REFERENCES auth.users(id) ON DELETE SET NULL,
  ADD COLUMN IF NOT EXISTS execution_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE public.ti_study_plan_versions
  ADD COLUMN IF NOT EXISTS input_dossier_hash TEXT,
  ADD COLUMN IF NOT EXISTS recommendation_hash TEXT,
  ADD COLUMN IF NOT EXISTS user_edit_hash TEXT,
  ADD COLUMN IF NOT EXISTS provider_model TEXT,
  ADD COLUMN IF NOT EXISTS deterministic_validation JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_ti_study_plan_versions_hash
  ON public.ti_study_plan_versions(recommendation_hash)
  WHERE recommendation_hash IS NOT NULL;

DROP POLICY IF EXISTS "TI members read Study plans" ON public.ti_study_plans;
CREATE POLICY "TI members read Study plans"
  ON public.ti_study_plans FOR SELECT TO authenticated
  USING (public.can_access_ti_workspace(workspace_id, NULL));

DROP POLICY IF EXISTS "TI editors update Study plans" ON public.ti_study_plans;
CREATE POLICY "TI editors update Study plans"
  ON public.ti_study_plans FOR UPDATE TO authenticated
  USING (public.can_access_ti_workspace(workspace_id, ARRAY['owner', 'admin', 'editor']))
  WITH CHECK (public.can_access_ti_workspace(workspace_id, ARRAY['owner', 'admin', 'editor']));

DROP POLICY IF EXISTS "TI members read Study plan versions" ON public.ti_study_plan_versions;
CREATE POLICY "TI members read Study plan versions"
  ON public.ti_study_plan_versions FOR SELECT TO authenticated
  USING (EXISTS (
    SELECT 1 FROM public.ti_study_plans AS plan
    WHERE plan.id = study_plan_id
      AND public.can_access_ti_workspace(plan.workspace_id, NULL)
  ));

DROP POLICY IF EXISTS "TI editors create Study plan versions" ON public.ti_study_plan_versions;
CREATE POLICY "TI editors create Study plan versions"
  ON public.ti_study_plan_versions FOR INSERT TO authenticated
  WITH CHECK (
    user_id = (SELECT auth.uid())
    AND EXISTS (
      SELECT 1 FROM public.ti_study_plans AS plan
      WHERE plan.id = study_plan_id
        AND public.can_access_ti_workspace(plan.workspace_id, ARRAY['owner', 'admin', 'editor'])
    )
  );

COMMENT ON COLUMN public.ti_study_plan_versions.deterministic_validation IS
  'Server-side compatibility result recorded beside the advisory AI recommendation.';
COMMENT ON COLUMN public.ti_study_plans.execution_snapshot IS
  'Exact user-confirmed plan snapshot used for collection/training gates.';

COMMIT;

