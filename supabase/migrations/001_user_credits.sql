-- MOPA Heightmap Studio — user credit tracking
-- Run via: supabase db push  (or paste into the Supabase SQL editor)

-- ── Table ──────────────────────────────────────────────────────────────────
create table if not exists public.user_credits (
  user_id    uuid        primary key references auth.users(id) on delete cascade,
  credits    integer     not null default 3,
  tier       text        not null default 'free',
  updated_at timestamptz not null default now()
);

-- ── Row-level security ─────────────────────────────────────────────────────
alter table public.user_credits enable row level security;

-- Authenticated users may read their own row only.
create policy "users_read_own_credits"
  on public.user_credits
  for select
  using (auth.uid() = user_id);

-- The service role (used by the FastAPI backend) bypasses RLS automatically.

-- ── Trigger: seed free credits on first sign-up ───────────────────────────
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.user_credits (user_id, credits, tier)
  values (new.id, 3, 'free')
  on conflict (user_id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- ── Atomic deduct function ────────────────────────────────────────────────
-- Returns new balance, or -1 if the user had fewer than p_amount credits.
create or replace function public.deduct_credit(
  p_user_id uuid,
  p_amount  integer default 1
)
returns integer
language plpgsql
security definer
as $$
declare
  v_balance integer;
begin
  select credits into v_balance
    from public.user_credits
   where user_id = p_user_id
     for update;

  if v_balance is null then
    -- Fallback: first-generation for a user whose row wasn't seeded.
    insert into public.user_credits (user_id, credits)
    values (p_user_id, 3)
    on conflict (user_id) do nothing;
    v_balance := 3;
  end if;

  if v_balance < p_amount then
    return -1;
  end if;

  update public.user_credits
     set credits    = credits - p_amount,
         updated_at = now()
   where user_id = p_user_id;

  return v_balance - p_amount;
end;
$$;

-- ── Atomic add function ───────────────────────────────────────────────────
-- Used by the Polar webhook handler to top up credits.
create or replace function public.add_credits(
  p_user_id uuid,
  p_amount  integer
)
returns integer
language plpgsql
security definer
as $$
begin
  insert into public.user_credits (user_id, credits)
  values (p_user_id, p_amount)
  on conflict (user_id) do update
    set credits    = public.user_credits.credits + excluded.credits,
        updated_at = now();

  return (select credits from public.user_credits where user_id = p_user_id);
end;
$$;
