-- Target Modeling Workspace V2 P5: immutable artifact registry and public
-- deployment metadata. Additive only; rollback disables public exposure and
-- retains ownership, artifacts, and deployment history.

BEGIN;

ALTER TABLE public.ti_model_artifacts
  ADD COLUMN IF NOT EXISTS checksum_sha256 TEXT,
  ADD COLUMN IF NOT EXISTS artifact_size_bytes BIGINT,
  ADD COLUMN IF NOT EXISTS manifest JSONB NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS storage_key TEXT,
  ADD COLUMN IF NOT EXISTS version UUID NOT NULL DEFAULT gen_random_uuid();

ALTER TABLE public.ti_model_deployments
  ADD COLUMN IF NOT EXISTS public_slug TEXT,
  ADD COLUMN IF NOT EXISTS visibility TEXT NOT NULL DEFAULT 'unlisted',
  ADD COLUMN IF NOT EXISTS display_name TEXT,
  ADD COLUMN IF NOT EXISTS description TEXT,
  ADD COLUMN IF NOT EXISTS scientific_notes TEXT,
  ADD COLUMN IF NOT EXISTS citations JSONB NOT NULL DEFAULT '[]'::jsonb,
  ADD COLUMN IF NOT EXISTS public_prediction_modes JSONB NOT NULL DEFAULT '["single", "bulk"]'::jsonb,
  ADD COLUMN IF NOT EXISTS allow_public_predictions BOOLEAN NOT NULL DEFAULT TRUE,
  ADD COLUMN IF NOT EXISTS allow_public_download BOOLEAN NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS public_requests_per_minute INTEGER NOT NULL DEFAULT 10,
  ADD COLUMN IF NOT EXISTS public_batch_limit INTEGER NOT NULL DEFAULT 100,
  ADD COLUMN IF NOT EXISTS owner_display_name TEXT,
  ADD COLUMN IF NOT EXISTS disabled_by UUID REFERENCES auth.users(id) ON DELETE SET NULL;

ALTER TABLE public.ti_model_deployments DROP CONSTRAINT IF EXISTS ti_model_deployments_visibility_check;
ALTER TABLE public.ti_model_deployments ADD CONSTRAINT ti_model_deployments_visibility_check CHECK (visibility IN ('unlisted', 'public'));
ALTER TABLE public.ti_model_deployments DROP CONSTRAINT IF EXISTS ti_model_deployments_slug_check;
ALTER TABLE public.ti_model_deployments ADD CONSTRAINT ti_model_deployments_slug_check CHECK (public_slug IS NULL OR public_slug ~ '^[a-z0-9][a-z0-9-]{2,79}$');
ALTER TABLE public.ti_model_deployments DROP CONSTRAINT IF EXISTS ti_model_deployments_public_limits_check;
ALTER TABLE public.ti_model_deployments ADD CONSTRAINT ti_model_deployments_public_limits_check CHECK (public_requests_per_minute BETWEEN 1 AND 120 AND public_batch_limit BETWEEN 1 AND 500);

ALTER TABLE public.ti_usage_events DROP CONSTRAINT IF EXISTS ti_usage_events_event_type_check;
ALTER TABLE public.ti_usage_events ADD CONSTRAINT ti_usage_events_event_type_check CHECK (event_type IN ('analysis_started', 'analysis_completed', 'analysis_failed', 'provider_request', 'chat_request', 'upload', 'export', 'public_prediction'));

CREATE UNIQUE INDEX IF NOT EXISTS idx_ti_model_deployments_owner_slug
  ON public.ti_model_deployments(user_id, public_slug)
  WHERE public_slug IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ti_model_deployments_public
  ON public.ti_model_deployments(public_slug, status)
  WHERE public_slug IS NOT NULL AND visibility IN ('public', 'unlisted');
CREATE UNIQUE INDEX IF NOT EXISTS idx_ti_model_artifacts_version ON public.ti_model_artifacts(version);

ALTER TABLE public.ti_model_deployments ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "TI members read Study deployments" ON public.ti_model_deployments;
CREATE POLICY "TI members read Study deployments"
  ON public.ti_model_deployments FOR SELECT TO authenticated
  USING (public.can_access_ti_workspace(workspace_id, NULL));
DROP POLICY IF EXISTS "TI owners manage Study deployments" ON public.ti_model_deployments;
CREATE POLICY "TI owners manage Study deployments"
  ON public.ti_model_deployments FOR ALL TO authenticated
  USING (public.can_access_ti_workspace(workspace_id, ARRAY['owner', 'admin']))
  WITH CHECK (public.can_access_ti_workspace(workspace_id, ARRAY['owner', 'admin']));

REVOKE ALL ON public.ti_model_deployments FROM anon;
GRANT SELECT ON public.ti_model_deployments TO authenticated;
GRANT INSERT, UPDATE ON public.ti_model_deployments TO authenticated;
REVOKE ALL ON public.ti_model_artifacts FROM anon;
REVOKE ALL ON public.ti_usage_events FROM anon;

DROP POLICY IF EXISTS "TI editors update Study datasets" ON public.ti_training_datasets;
CREATE POLICY "TI editors update Study datasets"
  ON public.ti_training_datasets FOR UPDATE TO authenticated
  USING (public.can_access_ti_workspace(workspace_id, ARRAY['owner', 'admin', 'editor']))
  WITH CHECK (public.can_access_ti_workspace(workspace_id, ARRAY['owner', 'admin', 'editor']));

COMMENT ON COLUMN public.ti_model_artifacts.storage_key IS 'Owner-scoped opaque storage key; never a filesystem path exposed to clients.';
COMMENT ON COLUMN public.ti_model_deployments.public_slug IS 'Shareable slug paired with deployment UUID ownership; username is display-only.';
COMMENT ON COLUMN public.ti_model_deployments.allow_public_download IS 'Owner-controlled public artifact download; default false.';

COMMIT;
