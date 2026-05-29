import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
// Dev-slice app shell. Boots straight into the Viewer on the local bundle,
// owns the engine Web Worker client + session state, and shows a Results
// panel when the session completes.
//
// The full screen flow (Landing / Consent / Registration / Tutorial /
// Computing) is PLAN §10 phase 4 — this slice exercises the engine end-to-end
// in the browser, which is the thing to verify first.
import { useEffect, useRef, useState } from "react";
import { Bundle } from "./bundle";
import { EngineClient } from "./engineClient";
import { Viewer } from "./components/Viewer";
import { COLORS, FONTS, IIIC_OPTIONS, VERDICT_STYLE } from "../ui/theme";
const BUNDLE_URL = "/bundle/v1.1-local";
export function App() {
    const [phase, setPhase] = useState("loading");
    const [bundle, setBundle] = useState(null);
    const [item, setItem] = useState(null);
    const [verdicts, setVerdicts] = useState([]);
    const [nQ, setNQ] = useState(0);
    const [msg, setMsg] = useState("");
    const clientRef = useRef(null);
    useEffect(() => {
        let disposed = false;
        Bundle.load(BUNDLE_URL)
            .then((b) => {
            if (disposed)
                return;
            setBundle(b);
            const client = new EngineClient({
                onItem: (it) => setItem(it),
                onDone: (r) => {
                    setVerdicts(r.verdicts);
                    setNQ(r.nQuestions);
                    setPhase("done");
                },
                onError: (m) => { setMsg(m); setPhase("error"); },
            });
            clientRef.current = client;
            client.start(b.inputs, `web-${Date.now()}`);
            setPhase("running");
        })
            .catch((e) => { setMsg(String(e)); setPhase("error"); });
        return () => {
            disposed = true;
            clientRef.current?.dispose();
        };
    }, []);
    if (phase === "loading")
        return _jsx(Centered, { children: "Loading test bank\u2026" });
    if (phase === "error")
        return _jsxs(Centered, { children: ["Error: ", msg] });
    if (phase === "done")
        return (_jsx(Centered, { children: _jsxs("div", { style: { width: 700 }, children: [_jsx("h1", { style: { fontFamily: FONTS.sans, color: COLORS.textPrimary, letterSpacing: 1 }, children: "Assessment Complete" }), _jsxs("div", { style: { color: COLORS.textBody, marginBottom: 16 }, children: [nQ, " recordings reviewed."] }), IIIC_OPTIONS.map((o, k) => {
                        const v = verdicts[k] ?? "PENDING";
                        const st = VERDICT_STYLE[v] ?? VERDICT_STYLE.PENDING;
                        return (_jsxs("div", { style: { display: "flex", justifyContent: "space-between",
                                padding: "8px 0", borderBottom: `1px solid ${COLORS.borderInactive}` }, children: [_jsx("span", { style: { color: COLORS.textSecondary, fontWeight: 600 }, children: o.label }), _jsx("span", { style: { color: st.color, fontWeight: 700 }, children: st.label })] }, o.code));
                    })] }) }));
    // running
    return (_jsx(Viewer, { bundle: bundle, item: item, onAnswer: (pick) => clientRef.current?.answer(pick) }));
}
function Centered({ children }) {
    return (_jsx("div", { style: { background: COLORS.bg, color: COLORS.textBody, fontFamily: FONTS.sans,
            height: "100vh", display: "flex", alignItems: "center", justifyContent: "center" }, children: children }));
}
