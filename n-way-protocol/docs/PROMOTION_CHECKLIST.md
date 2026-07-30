# Production promotion checklist

No item in this file authorizes a production change. Promotion requires a
separate explicit green light after every mandatory item is evidenced.

Post-freeze R&D has selected the artifact-ensemble plus Fisher-shortlist
combination for the next qualification round. This selection checks no box
below: the 2,000-replicate development study used a fixed 24-question frontier,
an unqualified artifact, and a research bank rather than the served-bank
Precision stopping workflow.

The isolated integration and a second 2,000-seed frozen-versus-integrated test
are complete. Relative coverage and tightening passed, but absolute skill and
bias coverage remained below 0.95. Accordingly, integration completion still
checks no production-promotion box below.

The integrated profile is now the standard for new isolated n-way work. The
frozen scalar profile is retained for replay/rollback. This internal standard
designation does not satisfy or waive any checklist item.

- [ ] DR07 scientific prerequisite passes on a new versioned fit.
- [ ] Response artifact is leakage-controlled, held-out calibrated, hashed,
      reviewed, and marked `qualified`.
- [ ] TypeScript/Python fixed-cloud and adaptive trajectory parity passes.
- [ ] Binary production golden suite remains unchanged.
- [ ] Skill SBC/recovery, RMSE, coverage, and width gates pass.
- [ ] Bias SBC/recovery, RMSE, coverage, and width gates pass.
- [ ] Full adaptive operating-characteristic decision gates pass.
- [ ] Selector-regret and production-bank latency gates pass.
- [ ] Particle/ESS/MH/MCSE profile is requalified and frozen.
- [ ] Serial and ranked-worker branches produce the same adopted state.
- [ ] Old binary sessions resume and finish byte-identically.
- [ ] N-way sessions reject profile/artifact drift on resume and result ingest.
- [ ] `isCorrect` is gold-derived; `matchedAskedTask` is a separate diagnostic.
- [ ] Server-owned rollout is off by default and unknown values fail closed.
- [ ] Offline shadow and internal test phases have reviewed reports.
- [ ] Allowlisted serial and ranked-worker phases meet observation gates.
- [ ] Rollback changes only new-session assignment and never changes an
      in-progress sitting's response model.
- [ ] Production code change is reviewed as a deliberate promotion from this
      directory, not copied ad hoc.
