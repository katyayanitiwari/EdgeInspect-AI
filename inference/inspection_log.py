"""Inspection logging to a local SQLite file (offline, no server).

Default database: logs/inspections.db. One row per inspected image/frame.
"""
import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "logs" / "inspections.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT NOT NULL,
    source          TEXT,
    weights         TEXT,
    decision        TEXT NOT NULL,
    severity        TEXT NOT NULL,
    defect_count    INTEGER NOT NULL,
    defects_json    TEXT NOT NULL,
    recommendation  TEXT,
    advisor_engine  TEXT,
    latency_ms      REAL,
    image_path      TEXT
)"""


def connect(path=DB_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(SCHEMA)
    return con


def log(con, report, advice, source="", weights="", image_path=""):
    cur = con.execute(
        "INSERT INTO inspections (timestamp, source, weights, decision, severity, defect_count, "
        "defects_json, recommendation, advisor_engine, latency_ms, image_path) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (time.strftime("%Y-%m-%d %H:%M:%S"), source, weights, report["decision"], report["severity"],
         report["defect_count"], json.dumps(report["defects"]), advice["text"], advice["engine"],
         report.get("latency_ms", {}).get("end_to_end"), image_path))
    con.commit()
    return cur.lastrowid

def recent(con, limit=50):
    cur = con.execute(
        "SELECT id, timestamp, source, decision, severity, defect_count, latency_ms "
        "FROM inspections ORDER BY id DESC LIMIT ?", (limit,))
    return [list(r) for r in cur.fetchall()]


def decision_counts(con):
    return dict(con.execute("SELECT decision, COUNT(*) FROM inspections GROUP BY decision").fetchall())
