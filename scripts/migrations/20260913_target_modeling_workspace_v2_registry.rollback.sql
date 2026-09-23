-- Operational rollback for P5: disable public access without deleting
-- artifacts, deployments, or model lineage.
BEGIN;
UPDATE public.ti_model_deployments
SET allow_public_predictions = FALSE,
    allow_public_download = FALSE,
    visibility = 'unlisted',
    updated_at = NOW();
COMMIT;
