\set ON_ERROR_STOP on

INSERT INTO auth.users (id, email) VALUES
  ('11111111-1111-4111-8111-111111111111', 'owner@example.invalid'),
  ('22222222-2222-4222-8222-222222222222', 'viewer@example.invalid'),
  ('33333333-3333-4333-8333-333333333333', 'outsider@example.invalid'),
  ('44444444-4444-4444-8444-444444444444', 'editor@example.invalid')
ON CONFLICT (id) DO NOTHING;

INSERT INTO public.profiles (id, role) VALUES
  ('11111111-1111-4111-8111-111111111111', 'user'),
  ('22222222-2222-4222-8222-222222222222', 'user'),
  ('33333333-3333-4333-8333-333333333333', 'user'),
  ('44444444-4444-4444-8444-444444444444', 'user')
ON CONFLICT (id) DO NOTHING;

INSERT INTO public.ti_workspaces (id, user_id, name, status) VALUES (
  'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
  '11111111-1111-4111-8111-111111111111',
  'P0 RLS fixture',
  'draft'
);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.ti_workspaces TO authenticated;
GRANT SELECT, INSERT, DELETE ON public.ti_workspace_targets TO authenticated;
GRANT SELECT ON public.ti_runs, public.ti_reports TO authenticated;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO authenticated;

SET ROLE authenticated;
SELECT set_config('request.jwt.claim.sub', '11111111-1111-4111-8111-111111111111', FALSE);

DO $$
BEGIN
  IF NOT public.can_access_ti_workspace('aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa', NULL) THEN
    RAISE EXCEPTION 'owner cannot access own Study';
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM public.ti_workspace_members
    WHERE workspace_id = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa'
      AND user_id = '11111111-1111-4111-8111-111111111111'
      AND role = 'owner'
  ) THEN
    RAISE EXCEPTION 'owner membership trigger did not run';
  END IF;
END;
$$;

INSERT INTO public.ti_workspace_members (
  workspace_id, user_id, role, invited_by, accepted_at
) VALUES (
  'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
  '22222222-2222-4222-8222-222222222222',
  'viewer',
  '11111111-1111-4111-8111-111111111111',
  NOW()
);

INSERT INTO public.ti_workspace_members (
  workspace_id, user_id, role, invited_by, accepted_at
) VALUES (
  'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
  '44444444-4444-4444-8444-444444444444',
  'editor',
  '11111111-1111-4111-8111-111111111111',
  NOW()
);

RESET ROLE;

INSERT INTO public.ti_runs (
  id, user_id, question, access_mode, status
) VALUES (
  'cccccccc-cccc-4ccc-8ccc-cccccccccccc',
  '44444444-4444-4444-8444-444444444444',
  'Editor-owned source run',
  'authenticated_only',
  'completed'
);

INSERT INTO public.ti_reports (
  id, run_id, user_id, report_version, report
) VALUES (
  'dddddddd-dddd-4ddd-8ddd-dddddddddddd',
  'cccccccc-cccc-4ccc-8ccc-cccccccccccc',
  '44444444-4444-4444-8444-444444444444',
  'fixture-v1',
  '{"targets":[{"canonical_name":"Fixture target","identifiers":{"uniprot":"P12345"}}]}'::JSONB
);

INSERT INTO public.ti_job_events (
  workspace_id, job_id, entity_type, event_type, status, progress_percent,
  safe_message, public_payload
) VALUES (
  'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
  'bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb',
  'training_run',
  'status_changed',
  'queued',
  0,
  'Training queued',
  '{"attempt": 1}'::JSONB
);

SET ROLE authenticated;
SELECT set_config('request.jwt.claim.sub', '22222222-2222-4222-8222-222222222222', FALSE);

DO $$
DECLARE
  affected INTEGER;
BEGIN
  IF NOT public.can_access_ti_workspace('aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa', NULL) THEN
    RAISE EXCEPTION 'accepted viewer cannot access Study';
  END IF;
  IF public.can_access_ti_workspace('aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa', ARRAY['editor']) THEN
    RAISE EXCEPTION 'viewer incorrectly received editor permission';
  END IF;
  IF (SELECT COUNT(*) FROM public.ti_job_events WHERE workspace_id = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa') <> 1 THEN
    RAISE EXCEPTION 'viewer cannot read safe Study events';
  END IF;
  UPDATE public.ti_workspaces
  SET name = 'Viewer must not edit'
  WHERE id = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';
  GET DIAGNOSTICS affected = ROW_COUNT;
  IF affected <> 0 THEN
    RAISE EXCEPTION 'viewer incorrectly updated Study';
  END IF;
END;
$$;

SELECT set_config('request.jwt.claim.sub', '44444444-4444-4444-8444-444444444444', FALSE);

UPDATE public.ti_workspaces
SET name = 'Editor updated Study'
WHERE id = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa';

INSERT INTO public.ti_workspace_targets (
  id, workspace_id, user_id, report_id, target_key, target_snapshot, added_by
) VALUES (
  'eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee',
  'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
  '11111111-1111-4111-8111-111111111111',
  'dddddddd-dddd-4ddd-8ddd-dddddddddddd',
  'P12345',
  '{"canonical_name":"Fixture target","identifiers":{"uniprot":"P12345"}}'::JSONB,
  '44444444-4444-4444-8444-444444444444'
);

INSERT INTO public.ti_workspace_target_sources (
  workspace_target_id, workspace_id, run_id, report_id,
  source_snapshot_hash, added_by, decision
) VALUES (
  'eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee',
  'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',
  'cccccccc-cccc-4ccc-8ccc-cccccccccccc',
  'dddddddd-dddd-4ddd-8ddd-dddddddddddd',
  repeat('a', 64),
  '44444444-4444-4444-8444-444444444444',
  'add_to_current'
);

SELECT set_config('request.jwt.claim.sub', '33333333-3333-4333-8333-333333333333', FALSE);

DO $$
BEGIN
  IF public.can_access_ti_workspace('aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa', NULL) THEN
    RAISE EXCEPTION 'outsider incorrectly received Study access';
  END IF;
  IF (SELECT COUNT(*) FROM public.ti_job_events WHERE workspace_id = 'aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa') <> 0 THEN
    RAISE EXCEPTION 'outsider can read Study events';
  END IF;
END;
$$;

RESET ROLE;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_publication_tables
    WHERE pubname = 'supabase_realtime'
      AND schemaname = 'public'
      AND tablename = 'ti_job_events'
  ) THEN
    RAISE EXCEPTION 'ti_job_events is not in the Realtime publication';
  END IF;
END;
$$;
