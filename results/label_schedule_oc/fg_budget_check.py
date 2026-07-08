"""Time-rescaling check: legacy sharp_static at evidence-equivalent budget
325 (240 * 172/127). If FG rises to ~0.67, the round-3b FG delta is the
fixed-window artifact of faster evidence flow, not degraded discrimination."""
import sys
sys.path.insert(0, '.')
from oc_campaign import run_session, ELL, SIG, np, SimBankAdapter, build_k7_engine_inputs

bank = SimBankAdapter(build_k7_engine_inputs())
fsi = np.exp(-(ELL - 0.10))
decl = []
for s in range(15):
    r = run_session(bank, weak_task=2, l_true=0.25, rule="static",
                    use_mixture=True, budget=325, seed=s, schedule=None,
                    filter_sigma_inf=fsi)
    decl.append(r["first_declared"] is not None)
    print(f"seed {s}: declared={r['first_declared']}", flush=True)
print(f"\nlegacy FG at budget 325: {np.mean(decl):.2f} (vs 0.60 at 240; randomized 0.67 at 240)")
