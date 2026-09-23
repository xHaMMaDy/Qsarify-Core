-- Target Modeling Workspace V2 P6: workspace-scoped audit history.
-- Additive and reversible operationally; rollback removes only the new audit
-- scope/policies and preserves the pre-existing global audit table.

BEGIN;

ALTER TABLE public.ti_audit_events
  ADD COLUMN IF NOT EXISTS workspace_id UUID REFERENCES public.ti_workspaces(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS idx_ti_audit_workspace_created
  ON public.ti_audit_events(workspace_id, created_at DESC)
  WHERE workspace_id IS NOT NULL;

DROP POLICY IF EXISTS "TI members read Study audit history" ON public.ti_audit_events;
CREATE POLICY "TI members read Study audit history"
  ON public.ti_audit_events FOR SELECT TO authenticated
  USING (workspace_id IS NOT NULL AND public.can_access_ti_workspace(workspace_id, NULL));

DROP POLICY IF EXISTS "TI members write Study audit history" ON public.ti_audit_events;
CREATE POLICY "TI members write Study audit history"
  ON public.ti_audit_events FOR INSERT TO authenticated
  WITH CHECK (
    workspace_id IS NOT NULL
    AND actor_user_id = (SELECT auth.uid())
    AND public.can_access_ti_workspace(workspace_id, ARRAY['owner', 'admin', 'editor', 'reviewer'])
  );

REVOKE ALL ON public.ti_audit_events FROM anon;
GRANT SELECT, INSERT ON public.ti_audit_events TO authenticated;

COMMENT ON COLUMN public.ti_audit_events.workspace_id IS
  'Optional Study scope for auditable workflow events; global legacy events remain supported.';

COMMIT;
