"""Tests for inference/severity.py. Uses fixed test rules, not the generated YAML,
so they keep passing when thresholds are recalibrated.

Run (project root):  .\.venv\Scripts\python.exe tests\test_severity.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.severity import assess, load_rules  # noqa: E402

RULES = {
    "confidence": {"min": 0.25, "review": 0.50},
    "count_escalation": 3,
    "decisions": {"none": "PASS", "minor": "PASS_WITH_NOTE", "major": "REWORK", "critical": "REJECT"},
    "classes": {"scratches": {"min_severity": "minor", "major_area": 10.0, "critical_area": 30.0}},
}
W = H = 10  # 10x10 image (100 px) -> box w*h in pixels == area %


def det(side, conf=0.9, cls="scratches"):
    return {"cls": cls, "conf": conf, "xyxy": [0, 0, side, side]}


def check(name, dets, severity, decision):
    r = assess(dets, W, H, RULES)
    assert (r["severity"], r["decision"]) == (severity, decision), f"{name}: got {r}"
    print(f"ok  {name:<40} {severity:<8} {decision}")


def main():
    check("no detections", [], "none", "PASS")
    check("below min confidence is ignored", [det(9, conf=0.2)], "none", "PASS")
    check("small defect (4%)", [det(2)], "minor", "PASS_WITH_NOTE")
    check("area exactly major threshold (10%)",
          [{"cls": "scratches", "conf": 0.9, "xyxy": [0, 0, 10, 1]}], "major", "REWORK")
    check("large defect (36%)", [det(6)], "critical", "REJECT")
    check("low-confidence minor -> review", [det(2, conf=0.4)], "minor", "MANUAL_REVIEW")
    check("low-confidence critical -> review", [det(6, conf=0.4)], "critical", "MANUAL_REVIEW")
    check("confident critical beats uncertain", [det(6), det(2, conf=0.4)], "critical", "REJECT")
    check("3 minor defects -> escalated to major", [det(2)] * 3, "major", "REWORK")
    try:
        assess([det(2, cls="unknown")], W, H, RULES)
        raise AssertionError("unknown class should raise KeyError")
    except KeyError:
        print("ok  unknown class raises KeyError")
    rules = load_rules()  # generated YAML must load and cover all 6 classes
    assert len(rules["classes"]) == 6, rules["classes"].keys()
    print("ok  severity_rules.yaml loads with 6 classes")
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
