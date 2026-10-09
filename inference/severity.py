"""Severity engine: detections -> per-defect severity -> part-level QC decision.

Rules (all numbers live in inference/severity_rules.yaml):
  1. Detections below confidence.min are ignored.
  2. Per defect: area% >= critical_area -> critical, >= major_area -> major,
     else the class's min_severity (default minor).
  3. Part severity = worst defect. If the part has >= count_escalation
     accepted defects, raise it one level (max critical).
  4. Decision: critical with confidence >= confidence.review -> REJECT.
     Otherwise, if any accepted defect has confidence < confidence.review
     -> MANUAL_REVIEW. Otherwise decisions[part severity].

Usage:
  from inference.severity import load_rules, assess
  report = assess(detections, img_w, img_h, load_rules())
where each detection is {"cls": "scratches", "conf": 0.81, "xyxy": [x1, y1, x2, y2]} in pixels.
"""
from pathlib import Path

import yaml

LEVELS = ["none", "minor", "major", "critical"]
RULES_PATH = Path(__file__).with_name("severity_rules.yaml")


def load_rules(path=RULES_PATH):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _defect_severity(area_pct, cls_rules):
    if area_pct >= cls_rules["critical_area"]:
        return "critical"
    if area_pct >= cls_rules["major_area"]:
        return "major"
    return cls_rules.get("min_severity", "minor")


def assess(detections, img_w, img_h, rules):
    conf_min = rules["confidence"]["min"]
    conf_review = rules["confidence"]["review"]
    img_area = float(img_w * img_h)

    defects = []
    for d in detections:
        if d["conf"] < conf_min:
            continue
        cls_rules = rules["classes"].get(d["cls"])
        if cls_rules is None:
            raise KeyError(f"No severity rule for class '{d['cls']}'")
        x1, y1, x2, y2 = d["xyxy"]
        area_pct = max(0.0, x2 - x1) * max(0.0, y2 - y1) / img_area * 100.0
        defects.append({
            "cls": d["cls"],
            "conf": round(float(d["conf"]), 3),
            "xyxy": [round(float(v), 1) for v in d["xyxy"]],
            "area_pct": round(area_pct, 2),
            "severity": _defect_severity(area_pct, cls_rules),
            "uncertain": d["conf"] < conf_review,
        })

    level = max((LEVELS.index(d["severity"]) for d in defects), default=0)
    reasons = []
    if defects:
        worst = max(defects, key=lambda d: (LEVELS.index(d["severity"]), d["area_pct"]))
        reasons.append(f"worst defect: {worst['cls']} ({worst['area_pct']}% of image, "
                       f"conf {worst['conf']}) -> {worst['severity']}")
    if len(defects) >= rules["count_escalation"] and level < LEVELS.index("critical"):
        level += 1
        reasons.append(f"{len(defects)} defects >= {rules['count_escalation']} -> raised to {LEVELS[level]}")
    severity = LEVELS[level]

    confident_critical = any(d["severity"] == "critical" and not d["uncertain"] for d in defects)
    if confident_critical:
        decision = rules["decisions"]["critical"]
    elif any(d["uncertain"] for d in defects):
        decision = "MANUAL_REVIEW"
        reasons.append(f"at least one defect has confidence < {conf_review}")
    else:
        decision = rules["decisions"][severity]
    if not defects:
        reasons.append("no defect above confidence threshold")

    return {"severity": severity, "decision": decision,
            "defect_count": len(defects), "defects": defects, "reasons": reasons}
