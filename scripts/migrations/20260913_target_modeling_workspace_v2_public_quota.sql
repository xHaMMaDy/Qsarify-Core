-- Target Modeling Workspace V2 P6: distributed public prediction quotas.
-- The bucket key contains only an irreversible caller hash, never an IP.

BEGIN;

CREATE TABLE IF NOT EXISTS public.ti_public_prediction_buckets (
  bucket_key TEXT NOT NULL,
  deployment_id UUID REFERENCES public.ti_model_deployments(id) ON DELETE CASCADE,
  window_started_at TIMESTAMPTZ NOT NULL,
  request_count INTEGER NOT NULL DEFAULT 0 CHECK (request_count >= 0),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (bucket_key, window_started_at)
);

CREATE INDEX IF NOT EXISTS idx_ti_public_prediction_buckets_window
  ON public.ti_public_prediction_buckets(window_started_at, deployment_id);

ALTER TABLE public.ti_public_prediction_buckets ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.ti_public_prediction_buckets FROM PUBLIC, anon, authenticated;

CREATE OR REPLACE FUNCTION public.consume_ti_public_prediction_quota(
  p_deployment_id UUID,
  p_caller_hash TEXT,
  p_deployment_limit INTEGER,
  p_global_limit INTEGER
)
RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
  current_window TIMESTAMPTZ := date_trunc('minute', NOW());
  caller_key TEXT := 'deployment:' || p_deployment_id::TEXT || ':caller:' || p_caller_hash;
  deployment_count INTEGER;
  global_count INTEGER;
BEGIN
  IF p_deployment_id IS NULL OR p_caller_hash !~ '^[0-9a-f]{16,128}$' OR p_deployment_limit < 1 OR p_global_limit < 1 THEN
    RETURN FALSE;
  END IF;
  -- One lock per minute protects both the deployment bucket and the global
  -- quota when several deployments are hit concurrently.
  PERFORM pg_advisory_xact_lock(hashtextextended(current_window::TEXT, 0));
  SELECT COALESCE(SUM(request_count), 0)::INTEGER INTO deployment_count
  FROM public.ti_public_prediction_buckets
  WHERE deployment_id = p_deployment_id AND window_started_at = current_window;
  SELECT COALESCE(SUM(request_count), 0)::INTEGER INTO global_count
  FROM public.ti_public_prediction_buckets
  WHERE window_started_at = current_window;
  IF deployment_count >= p_global_limit OR deployment_count >= p_deployment_limit OR global_count >= p_global_limit THEN
    RETURN FALSE;
  END IF;
  INSERT INTO public.ti_public_prediction_buckets(bucket_key, deployment_id, window_started_at, request_count)
  VALUES (caller_key, p_deployment_id, current_window, 1)
  ON CONFLICT (bucket_key, window_started_at)
  DO UPDATE SET request_count = public.ti_public_prediction_buckets.request_count + 1, updated_at = NOW();
  RETURN TRUE;
END;
$$;

REVOKE ALL ON FUNCTION public.consume_ti_public_prediction_quota(UUID, TEXT, INTEGER, INTEGER) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.consume_ti_public_prediction_quota(UUID, TEXT, INTEGER, INTEGER) TO service_role;

COMMENT ON FUNCTION public.consume_ti_public_prediction_quota(UUID, TEXT, INTEGER, INTEGER) IS
  'Atomic minute-window public prediction guard for the Next server proxy.';

COMMIT;
