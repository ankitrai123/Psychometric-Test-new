# Psychometric Assessment Platform

A candidate-facing online psychometric test built with React, TypeScript and Vite. It covers three stages:

1. **Registration & credentials**: guidelines, offer overview, document checklist (PAN, certificates, experience letters, pay slips, relieving letter, passport), and the registration window (opens/closes).
2. **Pre-test verification**: test metadata (question count, sections, duration), an automatic system compatibility check, and a rich-text input check where the candidate types `Hello World`.
3. **Assessment window**: live countdown, auto-save indicator, fullscreen, text-size settings, section dropdown, scrollable question palette, `Attempted: x/N`, Previous/Next, grid view, "Revisit Later" flags, a six-point Likert response panel, "Clear Response", and a Finish dialog with a summary.

| Registration | Verification |
| --- | --- |
| ![](docs/screenshots/1-registration.png) | ![](docs/screenshots/2-verification.png) |
| **Test window** | **Question grid** |
| ![](docs/screenshots/3-test-window.png) | ![](docs/screenshots/4-question-grid.png) |

## Getting started

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # typecheck + production build to dist/
```

## Replacing the questions

All test content is in [`src/data/test.json`](src/data/test.json), using this schema:

```json
{
  "testId": "psychometric_01",
  "title": "Personality Assessment",
  "durationMinutes": 45,
  "sections": [
    {
      "sectionId": "sec_1",
      "sectionTitle": "Personality Inventory",
      "questions": [
        {
          "id": 57,
          "questionText": "My performance is superior when working independently compared to collaborative settings.",
          "trait": "Extraversion",
          "options": ["Strongly Disagree", "Disagree", "Somewhat Disagree", "Somewhat Agree", "Agree", "Strongly Agree"]
        }
      ]
    }
  ]
}
```

- Question counts, section counts, duration and numbering all come from this file. Nothing else is hard-coded.
- Multiple sections are supported: they show up in the section dropdown and the grid view.
- `trait` is optional and passes through to the response export for scoring. Question `id`s must be unique across the whole test.
- The bundled data is 175 paraphrased personality statements in one section (45 minutes). They have no `trait` tags yet, so add them (and any reverse-keyed flags) before scoring.

Candidate profile, registration window, guidelines, document checklist and support contacts are in [`src/data/candidate.ts`](src/data/candidate.ts). The registration window defaults to "opened yesterday, closes in 7 days" so the demo always works. Replace it with real dates.

## State & auto-save

Session state (stage, documents confirmed, verification status, answers, revisit flags, current question, start time) lives in a reducer in [`src/hooks/useSession.ts`](src/hooks/useSession.ts). It is saved to `localStorage` under `psychometric:<testId>:session` on every change. After a refresh the candidate resumes where they left off, and the timer keeps counting from the original start time. When time runs out, the test is submitted automatically.

## Response export

After submission, the candidate can download a JSON file with one entry per question (`questionId`, `section`, `trait`, `optionIndex`, `optionText`, `markedForRevisit`), plus start/submit timestamps and the reason for submission. Analysis scripts consume this shape. When a backend exists, send the same payload from `CompletionPage` (`buildResponseExport`).

## Keyboard shortcuts (test window)

| Key | Action |
| --- | --- |
| `1`–`6` | Select option |
| `←` / `→` | Previous / next question |
| `Esc` | Close dialogs |

## Project layout

```
src/
  data/        test.json (questions), test.ts (flattening), candidate.ts (config)
  hooks/       useSession (reducer + persistence), useNow (ticking clock)
  pages/       RegistrationPage, VerificationPage, TestPage, CompletionPage
  components/  AppHeader, NotificationBanner, Footer, TestHeader, QuestionNav,
               QuestionGrid, FinishDialog, Modal, Brand, Icons
  styles.css   Design tokens + all styles
```

## Scoring backend

[`backend/`](backend/README.md) is a Python/FastAPI service. It scores submissions on 11 competency dimensions (Sten scores, response-quality checks, pre-written interpretations) and can add Claude-generated premium reports. It also stores results with encrypted responses, exports PDF, JSON or text, and serves an analytics dashboard at `/admin`.

To send candidates' answers to it, set `VITE_ASSESSMENT_API_URL` (see `.env.example`). The completion page then submits once and shows whether the server received the answers. Without it, the frontend runs offline as before.

## Candidate invitations

Admins create a personal test link for each candidate from the **Invite a candidate** card on `/admin` (name, optional email, candidate ID, role and how many days the link stays valid). The link looks like `https://<your-site>/?invite=<token>`:

- The candidate's own name, ID, email and role appear on their test; the registration window runs from when the link was created until it expires.
- Each link can be submitted once. Expired, cancelled, already-used and mistyped links show a clear message instead of the test.
- The Invitations table shows Invited, Started, Completed, Expired or Cancelled, with **Copy link**, **Cancel link** and **View report**.
- Set `REQUIRE_INVITE=true` on the backend so the test can only be taken through a personal link (the site then shows "Personal test link required" without one). Links point to `CANDIDATE_APP_URL` (defaults to the first non-localhost `CORS_ORIGINS` entry).

The admin dashboard asks for a username and password once `ADMIN_PASSWORD` is set on the backend (`ADMIN_USERNAME` defaults to `admin`). Sessions last 12 hours; changing the password signs everyone out.

Invites are stored in the backend database, so on serverless hosting (Vercel) connect a shared Postgres database via `DATABASE_URL`.

## Not yet included

There is no document upload or proctoring yet, and invite emails are sent from the admin's own mail app (the dashboard's **Email to candidate** button). The branding in `Brand.tsx` is a placeholder.
