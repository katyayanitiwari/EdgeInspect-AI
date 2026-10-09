"""QC advisor: severity report -> operator recommendation text.

PROTOTYPE STAGE: this is a rule-based template advisor. It will be replaced by an on-device
Llama model running through ExecuTorch (required by PS5). The interface stays the same:
    advise(report) -> {"text": str, "engine": str}
so the dashboard and logger do not change when Llama is plugged in.
"""

ENGINE = "rule-based template (placeholder for Llama + ExecuTorch)"

CLASS_NOTES = {
    "crazing": "network of fine surface cracks",
    "inclusion": "foreign material embedded in the surface",
    "patches": "irregular surface patch",
    "pitted_surface": "pitting (small cavities) on the surface",
    "rolled-in_scale": "oxide scale pressed into the surface during rolling",
    "scratches": "linear scratch marks",
}

ACTIONS = {
    "PASS": "No defect detected above the confidence threshold. Part can continue.",
    "PASS_WITH_NOTE": "Minor defect only. Part can continue; record it for trend monitoring.",
    "REWORK": "Major defect. Hold the part and send it for rework or supervisor inspection.",
    "REJECT": "Critical defect. Reject the part and quarantine it. If several rejects of the same "
              "defect type occur in a row, check the upstream process.",
    "MANUAL_REVIEW": "The model is not confident. A QC operator must inspect this part manually "
                     "before a decision is made.",
}


def advise(report):
    lines = [f"Decision: {report['decision']} (overall severity: {report['severity']})",
             ACTIONS[report["decision"]]]
    for i, d in enumerate(sorted(report["defects"], key=lambda d: -d["area_pct"]), 1):
        note = CLASS_NOTES.get(d["cls"], d["cls"])
        flag = " - low confidence, verify" if d["uncertain"] else ""
        ref = d.get("id", f"D{i:02d}")
        lines.append(f"{ref} — {d['cls']} ({note}): {d['severity']}, {d['area_pct']}% of image, "
                     f"confidence {d['conf']:.2f}{flag}")
    return {"text": "\n".join(lines), "engine": ENGINE}
