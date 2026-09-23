-- Target Modeling Workspace V2 P6: explicit account-deletion retention.
-- A request schedules application-data purge after 180 days. It never accepts
-- a client-supplied purge date and does not delete auth users from the web app.
BEGIN;

CREATE TABLE IF NOT EXISTS public.ti_account_deletion_requests (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL UNIQUE REFERENCES auth.users(id) ON DELETE CASCADE,
  requested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  purge_after TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '180 days'),
  status TEXT NOT NULL DEFAULT 'scheduled' CHECK (status IN ('scheduled', 'cancelled', 'executed', 'failed')),
  cancelled_at TIMESTAMPTZ,
  executed_at TIMESTAMPTZ,
  attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
  last_error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT ti_account_deletion_request_dates CHECK (purge_after >= requested_at)
);

CREATE INDEX IF NOT EXISTS idx_ti_account_deletion_due
  ON public.ti_account_deletion_requests(purge_after)
  WHERE status = 'scheduled';

ALTER TABLE public.ti_account_deletion_requests ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "TI users read own deletion request" ON public.ti_account_deletion_requests;
CREATE POLICY "TI users read own deletion request"
  ON public.ti_account_deletion_requests FOR SELECT TO authenticated
  USING (user_id = (SELECT auth.uid()));
DROP POLICY IF EXISTS "TI users create own deletion request" ON public.ti_account_deletion_requests;
CREATE POLICY "TI users create own deletion request"
  ON public.ti_account_deletion_requests FOR INSERT TO authenticated
  WITH CHECK (user_id = (SELECT auth.uid()) AND purge_after >= requested_at);
DROP POLICY IF EXISTS "TI users cancel own deletion request" ON public.ti_account_deletion_requests;
CREATE POLICY "TI users cancel own deletion request"
  ON public.ti_account_deletion_requests FOR UPDATE TO authenticated
  USING (user_id = (SELECT auth.uid()) AND status = 'scheduled')
  WITH CHECK (user_id = (SELECT auth.uid()) AND status = 'cancelled');

REVOKE ALL ON public.ti_account_deletion_requests FROM anon;
GRANT SELECT, INSERT, UPDATE ON public.ti_account_deletion_requests TO authenticated;

DROP TRIGGER IF EXISTS ti_account_deletion_requests_updated_at ON public.ti_account_deletion_requests;
CREATE TRIGGER ti_account_deletion_requests_updated_at
  BEFORE UPDATE ON public.ti_account_deletion_requests
  FOR EACH ROW EXECUTE FUNCTION public.handle_updated_at();

COMMENT ON TABLE public.ti_account_deletion_requests IS
  'Owner-requested application-data purge schedule; the retention window is fixed at 180 days.';

COMMIT;
