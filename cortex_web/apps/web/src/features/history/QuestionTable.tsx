import type { QuestionRow } from "../../api";

/** Participant-safe question history; AD6 rollback diagnostics stay server-side. */
export function QuestionTable({ rows }: { rows: QuestionRow[] }) {
  if (!rows.length) {
    return <div className="cx-placeholder">No per-question data for this attempt.</div>;
  }
  const num = (value: number | null, digits = 2) => (
    value === null ? "—" : value.toFixed(digits)
  );
  return (
    <div className="cx-q-scroll">
      <table className="cx-q-table">
        <thead>
          <tr>
            <th className="r">#</th>
            <th>Domain</th>
            <th>Your answer</th>
            <th>Correct</th>
            <th className="r">RT</th>
            <th className="r">ℓ / θ</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((question) => (
            <tr key={question.q}>
              <td className="r mono">{question.q}</td>
              <td>{question.domain}</td>
              <td className={question.isCorrect === null ? "" : question.isCorrect ? "ok" : "no"}>
                {question.answer ?? "—"}
              </td>
              <td>{question.correct ?? "—"}</td>
              <td className="r mono">
                {question.rt === null ? "—" : `${(question.rt / 1000).toFixed(1)}s`}
              </td>
              <td className="r mono">{num(question.ell)} / {num(question.theta)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
