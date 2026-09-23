-- Operational rollback: disable client access to the retention queue while
-- preserving scheduled requests and audit data for reconciliation.
BEGIN;
REVOKE ALL ON public.ti_account_deletion_requests FROM authenticated;
COMMIT;
