-- Account-owned uploaded model registry. Files are validated only by the
-- isolated worker; Next.js and Flask never deserialize them.

BEGIN;

CREATE TABLE IF NOT EXISTS public.ti_uploaded_models (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  workspace_id UUID REFERENCES public.ti_workspaces(id) ON DELETE SET NULL,
  display_name TEXT NOT NULL CHECK (char_length(display_name) BETWEEN 1 AND 160),
  storage_key TEXT NOT NULL UNIQUE,
  checksum_sha256 TEXT NOT NULL CHECK (checksum_sha256 ~ '^[0-9a-f]{64}$'),
  artifact_size_bytes BIGINT NOT NULL CHECK (artifact_size_bytes > 0),
  artifact_format TEXT NOT NULL CHECK (artifact_format IN ('pkl', 'joblib')),
  capability_profile JSONB NOT NULL DEFAULT '{}'::jsonb,
  manifest JSONB NOT NULL DEFAULT '{}'::jsonb,
  feature_schema JSONB NOT NULL DEFAULT '{}'::jsonb,
  target_semantics TEXT,
  metadata_confirmed BOOLEAN NOT NULL DEFAULT FALSE,
  status TEXT NOT NULL DEFAULT 'ready' CHECK (status IN ('inspecting', 'ready', 'disabled', 'failed')),
  last_error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ti_uploaded_models_user_created
  ON public.ti_uploaded_models(user_id, created_at DESC);

ALTER TABLE public.ti_uploaded_models ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "TI users manage own uploaded models" ON public.ti_uploaded_models;
CREATE POLICY "TI users manage own uploaded models"
  ON public.ti_uploaded_models FOR ALL TO authenticated
  USING (user_id = (SELECT auth.uid()))
  WITH CHECK (user_id = (SELECT auth.uid()));

REVOKE ALL ON public.ti_uploaded_models FROM anon;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.ti_uploaded_models TO authenticated;

DROP TRIGGER IF EXISTS ti_uploaded_models_updated_at ON public.ti_uploaded_models;
CREATE TRIGGER ti_uploaded_models_updated_at BEFORE UPDATE ON public.ti_uploaded_models
  FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

COMMENT ON COLUMN public.ti_uploaded_models.storage_key IS
  'Opaque owner-scoped key; never an absolute filesystem path or public URL.';
COMMENT ON COLUMN public.ti_uploaded_models.metadata_confirmed IS
  'Explicit owner confirmation of the detected feature schema and target semantics.';

COMMIT;
