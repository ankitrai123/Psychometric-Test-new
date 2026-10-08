import { StrictMode, type ReactNode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { API_URL } from './api';
import { applyInvite } from './data/candidate';
import { savedStage } from './hooks/useSession';
import { fetchInvite, inviteRequired, inviteTokenFromUrl } from './invite';
import { InviteGatePage } from './pages/InviteGatePage';
import './styles.css';

const root = createRoot(document.getElementById('root')!);
const show = (node: ReactNode) => root.render(<StrictMode>{node}</StrictMode>);

/**
 * Decides who is taking the test before the app renders:
 * - no backend configured: offline demo with the placeholder candidate
 * - ?invite=<token>: the candidate the admin invited (link must still be usable)
 * - no token: demo, unless the backend requires a personal link
 */
async function start() {
  if (!API_URL) return show(<App />);

  const token = inviteTokenFromUrl();
  show(<InviteGatePage reason="loading" />);
  if (!token) return show((await inviteRequired()) ? <InviteGatePage reason="missing" /> : <App />);

  const { details, reason } = await fetchInvite(token);
  if (!details) return show(<InviteGatePage reason={reason ?? 'invalid'} onRetry={() => void start()} />);

  applyInvite(token, details);
  const { status } = details;
  // A candidate who already submitted on this device still sees their confirmation page.
  if (status === 'invited' || status === 'started' || (status === 'completed' && savedStage() === 'completed')) {
    return show(<App />);
  }
  show(<InviteGatePage reason={status} name={details.name} />);
}

void start();
