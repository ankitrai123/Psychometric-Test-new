import { candidate, inviteState } from './data/candidate';
import type { SessionState } from './hooks/useSession';

/** Base URL of the scoring backend (backend/api_server.py). Unset = offline mode. */
export const API_URL: string | undefined = import.meta.env.VITE_ASSESSMENT_API_URL?.replace(/\/$/, '');

/** A submission the server refused for a reason retrying won't fix
 *  (e.g. the invite expired, was cancelled or was already used). */
export class SubmitError extends Error {
  constructor(public code: string, message: string) {
    super(message);
  }
}

/** Sends the candidate's answers for scoring. The frontend stores options
 *  0-based; the API expects Likert values 1..N. */
export async function submitAssessment(state: SessionState): Promise<string> {
  if (!API_URL) throw new Error('No API URL configured');
  const responses = Object.fromEntries(
    Object.entries(state.answers).map(([questionId, option]) => [questionId, option + 1]),
  );
  const res = await fetch(`${API_URL}/api/assess`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      test_taker_id: candidate.candidateId,
      name: candidate.name,
      email: candidate.email || null,
      responses,
      ...(inviteState.token ? { invite_token: inviteState.token } : {}),
    }),
  });
  if (!res.ok) {
    let detail: { code?: string; message?: string } | undefined;
    try {
      const body = await res.json();
      if (body && typeof body.detail === 'object') detail = body.detail;
    } catch {
      // Not JSON: treat as a transient failure.
    }
    if (detail?.code) throw new SubmitError(detail.code, detail.message ?? 'Submission was refused.');
    throw new Error(`Submission failed (${res.status})`);
  }
  const body: { assessment_id: string } = await res.json();
  return body.assessment_id;
}
