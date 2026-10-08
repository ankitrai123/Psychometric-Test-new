import { useCallback, useEffect, useRef, useState, type Dispatch } from 'react';
import { API_URL, SubmitError, submitAssessment } from '../api';
import { Check, Download } from '../components/Icons';
import { candidate, inviteState } from '../data/candidate';
import { questions, test } from '../data/test';
import type { SessionAction, SessionState } from '../hooks/useSession';

/** Builds the response payload an analysis/scoring script would consume. */
function buildResponseExport(state: SessionState) {
  return {
    testId: test.testId,
    title: test.title,
    candidate: { id: candidate.candidateId, name: candidate.name, email: candidate.email },
    startedAt: state.startedAt && new Date(state.startedAt).toISOString(),
    submittedAt: state.submittedAt && new Date(state.submittedAt).toISOString(),
    submitReason: state.submitReason,
    responses: questions.map((q) => {
      const option = state.answers[q.id];
      return {
        questionId: q.id,
        number: q.number,
        section: test.sections[q.sectionIndex].sectionId,
        trait: q.trait ?? null,
        questionText: q.questionText,
        optionIndex: option ?? null,
        optionText: option === undefined ? null : q.options[option],
        markedForRevisit: state.revisit.includes(q.id),
      };
    }),
  };
}

interface Props {
  state: SessionState;
  dispatch: Dispatch<SessionAction>;
}

type SyncStatus = 'offline' | 'sending' | 'sent' | 'failed' | 'rejected';

export function CompletionPage({ state, dispatch }: Props) {
  const attempted = Object.keys(state.answers).length;
  const [sync, setSync] = useState<SyncStatus>(
    !API_URL ? 'offline' : state.serverAssessmentId ? 'sent' : 'sending',
  );
  const [rejection, setRejection] = useState('');
  const inFlight = useRef(false);

  const send = useCallback(async () => {
    if (!API_URL || state.serverAssessmentId || inFlight.current) return;
    inFlight.current = true;
    setSync('sending');
    try {
      const assessmentId = await submitAssessment(state);
      dispatch({ type: 'serverAccepted', assessmentId });
      setSync('sent');
    } catch (err) {
      if (err instanceof SubmitError && err.code === 'completed') {
        // An earlier attempt reached the server even though we never saw the reply.
        dispatch({ type: 'serverAccepted', assessmentId: 'already-submitted' });
        setSync('sent');
      } else if (err instanceof SubmitError) {
        setRejection(err.message);
        setSync('rejected');
      } else {
        setSync('failed');
      }
    } finally {
      inFlight.current = false;
    }
  }, [state, dispatch]);

  useEffect(() => {
    void send();
    // Submit once on arrival; retries are manual.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const minutes =
    state.startedAt && state.submittedAt ? Math.max(1, Math.round((state.submittedAt - state.startedAt) / 60000)) : null;

  const download = () => {
    const blob = new Blob([JSON.stringify(buildResponseExport(state), null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${test.testId}_${candidate.candidateId}_responses.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main className="page page-complete">
      <section className="card complete-card">
        <span className="complete-badge"><Check size={32} /></span>
        <h1>Assessment submitted</h1>
        <p className="muted">
          {state.submitReason === 'timeout'
            ? 'Time ran out, so your responses were submitted automatically.'
            : 'Thank you. Your responses have been recorded.'}
        </p>
        <dl className="kv-list complete-stats">
          <div><dt>Test</dt><dd>{test.title}</dd></div>
          <div><dt>Answered</dt><dd>{attempted} of {questions.length}</dd></div>
          {minutes !== null && <div><dt>Time taken</dt><dd>{minutes} min</dd></div>}
        </dl>
        {sync === 'sending' && <p className="sync-note"><span className="spinner" /> Sending your responses…</p>}
        {sync === 'sent' && <p className="sync-note success-text">Responses received by the assessment server.</p>}
        {sync === 'failed' && (
          <p className="sync-note error-text">
            We couldn't reach the assessment server. Your answers are saved on this device.{' '}
            <button className="link-btn" onClick={() => void send()}>Retry</button>
          </p>
        )}
        {sync === 'rejected' && <p className="sync-note error-text">{rejection}</p>}
        <p className="muted">The recruitment team will contact you about next steps.</p>
        <div className="complete-actions">
          <button className="btn btn-outline" onClick={download}>
            <Download size={16} /> Download responses (JSON)
          </button>
          {!inviteState.token && (
            <button className="btn btn-ghost" onClick={() => dispatch({ type: 'reset' })}>
              Start over (demo)
            </button>
          )}
        </div>
      </section>
    </main>
  );
}
