// Computing-results spinner — shown briefly between the last answer and the
// results screen while the final verdicts are serialized and uploaded.

import { COLORS, FONTS } from "../../ui/theme";
import { Stage, Wordmark } from "./ui";

export function Computing({ note }: { note?: string }) {
  return (
    <Stage maxW={520}>
      <div style={{ textAlign: "center" }}>
        <Wordmark size={48} />
        <div style={{ margin: "32px auto 20px", width: 44, height: 44 }}>
          <div
            style={{
              width: 44,
              height: 44,
              border: `4px solid ${COLORS.borderInactive}`,
              borderTopColor: COLORS.accent,
              borderRadius: "50%",
              animation: "cortex-spin 0.9s linear infinite",
            }}
          />
        </div>
        <div style={{ fontFamily: FONTS.sans, color: COLORS.textBody, fontSize: 15 }}>
          {note || "Computing your results…"}
        </div>
        <style>{`@keyframes cortex-spin { to { transform: rotate(360deg); } }`}</style>
      </div>
    </Stage>
  );
}
