-- Operational rollback: stop new quota calls without deleting usage history.
BEGIN;
REVOKE EXECUTE ON FUNCTION public.consume_ti_public_prediction_quota(UUID, TEXT, INTEGER, INTEGER) FROM service_role;
COMMIT;
