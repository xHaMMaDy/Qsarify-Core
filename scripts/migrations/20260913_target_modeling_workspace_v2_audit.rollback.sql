-- Operational rollback: disable the new Study-audit writes/read policy. The
-- additive column is retained so existing audit records are never destroyed.
BEGIN;
DROP POLICY IF EXISTS "TI members read Study audit history" ON public.ti_audit_events;
DROP POLICY IF EXISTS "TI members write Study audit history" ON public.ti_audit_events;
COMMIT;
