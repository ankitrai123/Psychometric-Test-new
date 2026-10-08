import { Brand } from '../components/Brand';
import { Footer } from '../components/Footer';
import { Check, Clock, Info, X } from '../components/Icons';
import { support } from '../data/candidate';
import type { GateReason } from '../invite';

const MESSAGES: Record<GateReason, { title: string; body: string }> = {
  missing: {
    title: 'Personal test link required',
    body: 'This assessment is by invitation only. Please open the personal link from your invitation email.',
  },
  invalid: {
    title: 'This link is not valid',
    body: 'Please check that you opened the complete link from your invitation email, or ask the recruitment team to send it again.',
  },
  expired: {
    title: 'This link has expired',
    body: 'The time window for this assessment has closed. Please contact the recruitment team if you need a new link.',
  },
  revoked: {
    title: 'This link has been cancelled',
    body: 'This invitation is no longer active. Please contact the recruitment team for help.',
  },
  completed: {
    title: 'Assessment already submitted',
    body: 'Your responses for this invitation have already been received. Each link can only be used once.',
  },
  network: {
    title: "We couldn't reach the assessment server",
    body: 'Please check your internet connection and try again.',
  },
};

interface Props {
  reason: GateReason | 'loading';
  name?: string;
  onRetry?: () => void;
}

/** Shown instead of the test when there is no usable invite link. */
export function InviteGatePage({ reason, name, onRetry }: Props) {
  if (reason === 'loading') {
    return (
      <div className="app-shell">
        <header className="app-header"><Brand /></header>
        <main className="page page-complete">
          <p className="sync-note muted"><span className="spinner" /> Checking your test link…</p>
        </main>
      </div>
    );
  }
  const { title, body } = MESSAGES[reason];
  const Icon =
    reason === 'expired' ? Clock : reason === 'completed' ? Check : reason === 'missing' || reason === 'network' ? Info : X;
  return (
    <div className="app-shell">
      <header className="app-header"><Brand /></header>
      <main className="page page-complete">
        <section className="card complete-card">
          <span className={`complete-badge gate-badge gate-${reason}`}><Icon size={30} /></span>
          <h1>{title}</h1>
          {name && <p className="muted">Invitation for {name}</p>}
          <p>{body}</p>
          {reason === 'network' && onRetry && (
            <button className="btn btn-primary" onClick={onRetry}>Try again</button>
          )}
          <p className="muted gate-help">
            Need help? <a href={`mailto:${support.email}`}>{support.email}</a> · {support.phone}
          </p>
        </section>
      </main>
      <Footer />
    </div>
  );
}
