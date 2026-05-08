import { createClient, SupabaseClient } from '@supabase/supabase-js';

function readMeta(name: string): string {
  return (
    document.querySelector<HTMLMetaElement>(`meta[name="${name}"]`)?.content ?? ''
  );
}

const url = readMeta('supabase-url');
const anonKey = readMeta('supabase-anon-key');

const configured =
  url.length > 0 &&
  url !== '__SUPABASE_URL__' &&
  anonKey.length > 0 &&
  anonKey !== '__SUPABASE_ANON_KEY__';

export const supabase: SupabaseClient = configured
  ? createClient(url, anonKey)
  : createClient('https://placeholder.supabase.co', 'placeholder-anon-key');

export const supabaseConfigured = configured;
