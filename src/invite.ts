import { API_URL } from './api';
import type { InviteDetails } from './data/candidate';

/** Why the test can't be shown: no link, a bad link, or the server is unreachable. */
export type GateReason = 'missing' | 'invalid' | 'expired' | 'revoked' | 'completed' | 'network';

/** The ?invite=<token> value from the address bar, if it looks valid. */
export function inviteTokenFromUrl(): string | null {
  const token = new URLSearchParams(window.location.search).get('invite');
  return token && /^[A-Za-z0-9_-]{8,64}$/.test(token) ? token : null;
}

export async function fetchInvite(token: string): Promise<{ details?: InviteDetails; reason?: GateReason }> {
  try {
    const res = await fetch(`${API_URL}/api/invites/${encodeURIComponent(token)}`);
    if (res.status === 404) return { reason: 'invalid' };
    if (!res.ok) return { reason: 'network' };
    return { details: (await res.json()) as InviteDetails };
  } catch {
    return { reason: 'network' };
  }
}

/** Whether the backend only accepts candidates with a personal link. */
export async function inviteRequired(): Promise<boolean> {
  try {
    const res = await fetch(`${API_URL}/health`);
    return res.ok && Boolean((await res.json()).invite_required);
  } catch {
    return false;
  }
}
