import { candidate } from '../data/candidate';

/** Organization logo (public/pasona-logo.png) next to the product name. */
export function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <div className="brand">
      <img
        className={`brand-logo${compact ? ' brand-logo-compact' : ''}`}
        src="/pasona-logo.png"
        alt={candidate.organization}
        width={469}
        height={96}
      />
      {!compact && (
        <>
          <span className="brand-divider" aria-hidden="true" />
          <span className="brand-partner">
            <strong>Psych</strong>Assess
          </span>
        </>
      )}
    </div>
  );
}
