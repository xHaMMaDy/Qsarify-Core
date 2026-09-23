-- Operational rollback: disable the upload registry while preserving files for
-- a separate owner-approved cleanup. No automatic artifact deletion here.
BEGIN;
REVOKE ALL ON public.ti_uploaded_models FROM authenticated;
COMMIT;
