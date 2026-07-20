import { useEffect, useState } from "react";

import * as api from "../../api";
import { VerdictChip } from "../dashboard/verdict";

/** Read-only view of the participant's active spaced-repetition protocol. */
export function ProtocolSurface() {
  const [regimen, setRegimen] = useState<api.RegimenPlan | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState(false);

  useEffect(() => {
    let live = true;
    api.getRegimen()
      .then((result) => {
        if (live) {
          setRegimen(result.regimen);
          setLoaded(true);
        }
      })
      .catch(() => { if (live) setError(true); });
    return () => { live = false; };
  }, []);

  return (
    <section className="cx-panel" aria-label="My protocol">
      <div className="cx-phead"><h2>My protocol</h2></div>
      {error && !regimen && (
        <div className="cx-placeholder">Could not load your protocol. Try refreshing.</div>
      )}
      {loaded && !regimen && (
        <div className="cx-placeholder">
          No protocol yet. Your first training session builds a spaced-repetition plan from your certification result, weighted toward the domains still below their ℓ* bar.
        </div>
      )}
      {regimen && (
        <>
          <div className="cx-weekhdr">
            <span className="big">Week {regimen.weekOf} of {regimen.weeks}</span>
            <span className="sub">spaced-repetition plan toward each task&apos;s target ℓ*</span>
          </div>
          <div className="cx-weekbar" aria-label={`Week ${regimen.weekOf} of ${regimen.weeks}`}>
            {Array.from({ length: regimen.weeks }, (_value, index) => {
              const week = index + 1;
              const className = week < regimen.weekOf ? "done" : week === regimen.weekOf ? "now" : "";
              return <span key={week} className={`wk ${className}`} />;
            })}
          </div>
          <table className="cx-deck">
            <thead>
              <tr>
                <th className="task">Task</th>
                <th className="r">ℓ → ℓ*</th>
                <th className="cnt">New</th>
                <th className="cnt">Learning</th>
                <th className="cnt">Due</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {(regimen.deck ?? []).map((row) => {
                const reached = row.ell >= row.ellStar;
                return (
                  <tr key={row.code}>
                    <td className="task">{row.label}</td>
                    <td className="ellcell r">
                      {row.ell.toFixed(2)} <span className="of">→ {row.ellStar.toFixed(2)}</span>
                    </td>
                    <td className={`cnt new${row.new ? "" : " zero"}`}>{row.new}</td>
                    <td className={`cnt learn${row.learning ? "" : " zero"}`}>{row.learning}</td>
                    <td className={`cnt due${row.due ? "" : " zero"}`}>{row.due}</td>
                    <td><VerdictChip verdict={reached ? "PASS" : "IN_TRAINING"} /></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}
