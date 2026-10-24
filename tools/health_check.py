"""Hourly health check of the live system: metrics with their change, and what is "not OK".

The deterministic half of the health-watch skill (.claude/skills/health-watch): it only
reads what the system already produces (pulled metrics, the synced DB, status and log)
and decides whether a Claude investigation is needed. "OK" = every check below passes
for the window [since, until) (default: since the previous run, else the last hour).
Alerts (any one -> not OK, a Claude investigation):

Running
- process_down / process_restart: no app process, or its pid changed since the previous run
- fps_low: a perf minute with FPS < 18 (the health signal; loop time grows with people)
- queue_backlog: inference queue > 10
- log_errors: ERROR / WARNING lines in the window (known noise filtered: KNOWN_LOG_NOISE)
- disk_low: < 20 GB free on the Jetson
Identity
- contradiction: tools/find_contradictions.py on the window's audit samples
- gallery_cross_person: >= GALLERY_MIN_SAMPLES new Re-ID gallery samples of one track closer to
  ANOTHER person's gallery (mean of the 3 best cosines, margin GALLERY_MARGIN) — 2026-10-23:
  20 Virag crops in Ati's (margin 0.06-0.17)
- face_name_age_revoked: the age gate revoked a face-confirmed name (rolling-window risk, Virag)
Trend (rolling 24 h vs the earlier days, once >= 3 days of naming records exist)
- naming_drop: named share fell > 10 points / median time to name grew > 1.5x

Info (counted in the metrics line, no investigation; INFO_KINDS): long_unnamed (an ended
track >= LONG_UNNAMED_SEC never named: back to the camera, ghost), switch_revoked
(exclusivity unnamed a switched track: handled), loop_slow (loop avg > 55 ms).
Backtest 2026-10-23 04-11 UTC: these were 3-5 per busy hour, mostly explained.

Every run appends one metrics line (with deltas vs the previous run) to
local/health/<local day>.md, the full record to local/health/<local day>.jsonl, and
prints JSON findings that were not reported before (dedup key per finding).
Exit code: 0 = all OK or nothing new, 1 = new findings (investigate).

Usage:
    scripts/remote/health_check.sh                       # pulls, then runs this
    PYTHONPATH=. python tools/health_check.py local/remote/metrics --db local/remote/database.db \\
        --config local/remote/config.jetson-live.yaml [--since ISO --until ISO] [--no-remote] [--dry-run]
"""

import argparse
import json
import re
import sqlite3
import subprocess
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from tools.analyze_naming import summarize
from tools.audit_sheet import load_samples
from tools.find_contradictions import find, load_face_attempts

FPS_MIN = 18.0
LOOP_MAX_MS = 55.0
QUEUE_MAX = 10
DISK_MIN_GB = 20.0
GALLERY_MARGIN = 0.05
GALLERY_MIN_SAMPLES = 3
LONG_UNNAMED_SEC = 180.0
MIN_TRACK_SEC = 5.0
KNOWN_LOG_NOISE = ("VLM call failed", "cannot schedule new futures after shutdown")
INFO_KINDS = {"long_unnamed": "névtelen≥3p", "switch_revoked": "csere", "loop_slow": "lassú ciklus"}
PERSONS = {1: "Ati", 2: "Judit", 3: "Virag", 4: "Rozika"}
HEALTH_DIR = Path("local/health")


def _ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # partially written last line
    return rows


def _days(since: datetime, until: datetime) -> list[str]:
    d, days = since.astimezone(timezone.utc).date(), []
    while d <= until.astimezone(timezone.utc).date():
        days.append(d.isoformat())
        d += timedelta(days=1)
    return days


def _stream(metrics: Path, name: str, since: datetime, until: datetime) -> list[dict]:
    """Records of a daily metrics stream with since <= ts < until."""
    lo, hi = _ts(since), _ts(until)
    return [r for day in _days(since, until) for r in _jsonl(metrics / name / f"{day}.jsonl")
            if lo <= r["ts"] < hi]


def perf(metrics: Path, since: datetime, until: datetime) -> tuple[dict, list[dict]]:
    rows = _stream(metrics, "perf", since, until)
    if not rows:
        return {}, []
    m = {"fps": round(float(np.mean([r["fps"] for r in rows])), 1),
         "fps_min": round(min(r["fps"] for r in rows), 1),
         "loop_ms": round(float(np.mean([r["loop_ms"] for r in rows])), 1),
         "queue_max": max(r.get("inference_queue", 0) for r in rows)}
    findings = []
    slow = [r for r in rows if r["fps"] < FPS_MIN]
    if slow:
        findings.append({"kind": "fps_low", "key": f"fps_low:{slow[0]['ts'][:13]}", "minutes": len(slow),
                         "first": slow[0]["ts"], "min": m["fps_min"]})
    if m["loop_ms"] > LOOP_MAX_MS:
        findings.append({"kind": "loop_slow", "key": f"loop_slow:{rows[0]['ts'][:13]}", "loop_ms": m["loop_ms"]})
    if m["queue_max"] > QUEUE_MAX:
        findings.append({"kind": "queue_backlog", "key": f"queue:{rows[0]['ts'][:13]}", "max": m["queue_max"]})
    return m, findings


def naming(metrics: Path, since: datetime, until: datetime) -> tuple[dict, list[dict]]:
    """Tracks that ENDED in the window (record ts = track start, + duration)."""
    rows = []
    for r in _stream(metrics, "naming", since - timedelta(days=1), until):
        end = datetime.fromisoformat(r["ts"]) + timedelta(seconds=r["duration_sec"])
        if since <= end < until and r["duration_sec"] >= MIN_TRACK_SEC:
            rows.append(r)
    s = summarize(rows)
    m = {"tracks": s["tracks"], "named": round(s["named"], 2),
         "t50": None if s["t50"] is None else round(s["t50"], 1),
         "t90": None if s["t90"] is None else round(s["t90"], 1), "unstable": round(s["unstable"], 2)}
    findings = [{"kind": "long_unnamed", "key": f"long_unnamed:{r['track']}", "track": r["track"],
                 "camera": r["camera"], "start": r["ts"], "duration_sec": r["duration_sec"]}
                for r in rows if r["named_after_sec"] is None and r["duration_sec"] >= LONG_UNNAMED_SEC]
    return m, findings


def naming_trend(metrics: Path, until: datetime) -> tuple[dict, list[dict]]:
    """Last 24 h vs the days before (needs >= 3 earlier days of records)."""
    day = timedelta(days=1)
    last, _ = naming(metrics, until - day, until)
    earlier = [naming(metrics, until - (k + 1) * day, until - k * day)[0] for k in range(1, 8)]
    earlier = [e for e in earlier if e["tracks"]]
    m = {"named_24h": last["named"], "t50_24h": last["t50"]}
    if len(earlier) < 3 or not last["tracks"]:
        return m, []
    named_base = float(np.mean([e["named"] for e in earlier]))
    t50s = [e["t50"] for e in earlier if e["t50"] is not None]
    m.update(named_base=round(named_base, 2), t50_base=round(float(np.median(t50s)), 1) if t50s else None)
    findings = []
    if last["named"] < named_base - 0.10 or (last["t50"] and m["t50_base"] and last["t50"] > 1.5 * m["t50_base"]):
        findings.append({"kind": "naming_drop", "key": f"naming_drop:{until.date()}", **m})
    return m, findings


def contradictions(metrics: Path, since: datetime, until: datetime, config: Optional[Path]) -> list[dict]:
    zone_manager = None
    if config:
        from src.config import load_config
        from src.tracking.zone_manager import ZoneManager
        zone_manager = ZoneManager(load_config(str(config)).zones)
    samples = load_samples(metrics, _ts(since), _ts(until), None)
    out = []
    for f in find(samples, load_face_attempts(metrics, _ts(since), PERSONS), zone_manager):
        tracks = ",".join(sorted(str(t) for t in f.get("tracks", []) or [f.get("track", "")]))
        out.append({**f, "kind": "contradiction", "type": f["kind"], "key": f"contradiction:{f['kind']}:{tracks}"})
    return out


def age_revocations(metrics: Path, since: datetime, until: datetime) -> list[dict]:
    return [{"kind": "face_name_age_revoked", "key": f"age_revoked:{r['track_id']}", "track": r["track_id"],
             "camera": r.get("camera_id"), "person": PERSONS.get(r["person_id"]), "ts": r["ts"],
             "track_age": r["track_age"], "person_age": r["person_age"]}
            for r in _stream(metrics, "age", since, until)
            if r.get("type") == "veto" and r.get("action") == "revoked" and r.get("method") == "face"]


def db_checks(db: Path, since: datetime, until: datetime) -> list[dict]:
    """Exclusivity revocations and cross-person Re-ID gallery samples (DB timestamps: naive UTC)."""
    lo = since.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    hi = until.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    findings = []
    for ts, track, camera, person, extra in con.execute(
            "SELECT timestamp, track_id, camera_id, person_id, extra_data FROM events "
            "WHERE event_type = 'id_revoked' AND timestamp >= ? AND timestamp < ?", (lo, hi)):
        reason = (json.loads(extra) if extra else {}).get("reason", "")
        if "face-confirmed on" in reason:
            findings.append({"kind": "switch_revoked", "key": f"switch:{track}", "track": track, "camera": camera,
                             "person": PERSONS.get(person), "ts": ts, "reason": reason})
    rows = con.execute("SELECT id, person_id, track_id, timestamp, embedding FROM reid_embeddings").fetchall()
    con.close()
    unit = lambda b: (v := np.frombuffer(b, np.float32)) / (np.linalg.norm(v) + 1e-9)
    new = [r for r in rows if lo <= r[3] < hi]
    old = [r for r in rows if r[3] < lo]
    refs = {p: np.stack([unit(r[4]) for r in old if r[1] == p]) for p in {r[1] for r in old}}
    bad = Counter()
    for r in new:
        if r[1] not in refs:
            continue
        v = unit(r[4])
        score = {p: float(np.sort(m @ v)[-3:].mean()) for p, m in refs.items()}
        other = max((p for p in score if p != r[1]), key=score.get, default=None)
        if other is not None and score[other] > score[r[1]] + GALLERY_MARGIN:
            bad[(r[2], r[1], other)] += 1
    for (track, person, other), n in bad.items():
        if n < GALLERY_MIN_SAMPLES:
            continue
        findings.append({"kind": "gallery_cross_person", "key": f"gallery:{track}:{person}", "track": track,
                         "gallery": PERSONS.get(person), "closer_to": PERSONS.get(other), "samples": n})
    return findings


def remote(since: datetime, state: dict) -> tuple[dict, list[dict]]:
    """App pid, disk and log lines in the window (scripts/remote; log times are Jetson local)."""
    status = subprocess.run(["scripts/remote/rstatus.sh"], capture_output=True, text=True, timeout=120).stdout
    pid = (re.search(r"RUNNING \(pid (\d+)", status) or [None, None])[1]
    disk = re.search(r"(\d+)G free of", status)
    m = {"pid": pid, "disk_free_gb": int(disk[1]) if disk else None}
    findings = []
    if pid is None:
        findings.append({"kind": "process_down", "key": f"down:{_ts(since)[:13]}", "status": status[-500:]})
    elif state.get("pid") and pid != state["pid"]:
        findings.append({"kind": "process_restart", "key": f"restart:{pid}", "pid": pid, "previous": state["pid"]})
    if m["disk_free_gb"] is not None and m["disk_free_gb"] < DISK_MIN_GB:
        findings.append({"kind": "disk_low", "key": f"disk:{_ts(since)[:10]}", "free_gb": m["disk_free_gb"]})
    log = subprocess.run(["scripts/remote/rlog.sh", "errors", "2000"], capture_output=True, text=True,
                         timeout=120).stdout
    lo = since.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    lines = [l for l in log.splitlines() if l[:19] >= lo and not any(n in l for n in KNOWN_LOG_NOISE)
             and not re.search(r"rtsp|pass", l, re.I)]
    m["log_lines"] = len(lines)
    if lines:
        kinds = Counter(re.sub(r"\d+", "N", l[24:])[:120] for l in lines)
        findings.append({"kind": "log_errors", "key": "log:" + "|".join(sorted(kinds))[:200],
                         "count": len(lines), "examples": [k for k, _ in kinds.most_common(5)]})
    return m, findings


def _delta(now, before) -> str:
    if now is None or before is None:
        return ""
    d = now - before
    return f" ({d:+.1f})" if isinstance(d, float) else f" ({d:+d})"


def summary_line(until: datetime, m: dict, prev: dict, ok: bool, n_new: int, info: Counter) -> str:
    p, n, t = m.get("perf", {}), m.get("naming", {}), m.get("trend", {})
    pp, pn = prev.get("perf", {}), prev.get("naming", {})
    fmt = lambda v, unit="": "-" if v is None else f"{v}{unit}"
    parts = [f"{until.astimezone():%H:%M} {'✓' if ok else '✗ ' + str(n_new) + ' új'}",
             f"FPS {fmt(p.get('fps'))}{_delta(p.get('fps'), pp.get('fps'))} min {fmt(p.get('fps_min'))}",
             f"loop {fmt(p.get('loop_ms'), ' ms')}{_delta(p.get('loop_ms'), pp.get('loop_ms'))}",
             f"névadás: {n.get('tracks', 0)} track, {round(100 * n.get('named', 0))}%"
             f"{_delta(round(100 * n.get('named', 0)), round(100 * pn['named']) if 'named' in pn else None)}, "
             f"t50 {fmt(n.get('t50'), ' s')}{_delta(n.get('t50'), pn.get('t50'))}",
             f"24h: {round(100 * (t.get('named_24h') or 0))}%, t50 {fmt(t.get('t50_24h'), ' s')}"]
    if info:
        parts.append("infó: " + ", ".join(f"{n} {INFO_KINDS[k]}" for k, n in info.items()))
    return " | ".join(parts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("metrics", type=Path)
    p.add_argument("--db", type=Path, default=Path("local/remote/database.db"))
    p.add_argument("--config", type=Path)
    p.add_argument("--since")
    p.add_argument("--until")
    p.add_argument("--no-remote", action="store_true", help="skip status/log (backtests)")
    p.add_argument("--dry-run", action="store_true", help="do not write local/health or the state")
    args = p.parse_args()

    state_path = HEALTH_DIR / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    until = datetime.fromisoformat(args.until) if args.until else datetime.now(timezone.utc)
    since = datetime.fromisoformat(args.since) if args.since else \
        datetime.fromisoformat(state["until"]) if state.get("until") else until - timedelta(hours=1)
    until, since = until.astimezone(timezone.utc), since.astimezone(timezone.utc)

    metrics, findings = {}, []
    metrics["perf"], f = perf(args.metrics, since, until); findings += f
    metrics["naming"], f = naming(args.metrics, since, until); findings += f
    metrics["trend"], f = naming_trend(args.metrics, until); findings += f
    findings += contradictions(args.metrics, since, until, args.config)
    findings += age_revocations(args.metrics, since, until)
    if args.db.exists():
        findings += db_checks(args.db, since, until)
    if not args.no_remote:
        metrics["remote"], f = remote(since, state); findings += f

    info = Counter(f["kind"] for f in findings if f["kind"] in INFO_KINDS)
    alerts = list({f["key"]: f for f in findings if f["kind"] not in INFO_KINDS}.values())
    reported = set(state.get("reported", []))
    new = [f for f in alerts if f["key"] not in reported]
    line = summary_line(until, metrics, state.get("metrics", {}), not alerts, len(new), info)
    print(line)
    for f in new:
        print(json.dumps(f, ensure_ascii=False, default=str))

    if not args.dry_run:
        HEALTH_DIR.mkdir(parents=True, exist_ok=True)
        day = until.astimezone().date().isoformat()
        with open(HEALTH_DIR / f"{day}.md", "a") as fh:
            fh.write(line + "\n")
        with open(HEALTH_DIR / f"{day}.jsonl", "a") as fh:
            fh.write(json.dumps({"since": _ts(since), "until": _ts(until), "metrics": metrics,
                                 "alerts": alerts, "info": dict(info), "new": [f["key"] for f in new]}, default=str) + "\n")
        state.update(until=_ts(until), metrics=metrics, pid=metrics.get("remote", {}).get("pid") or state.get("pid"),
                     reported=sorted(reported | {f["key"] for f in alerts})[-2000:])
        state_path.write_text(json.dumps(state, indent=1, default=str))
    raise SystemExit(1 if new else 0)


if __name__ == "__main__":
    main()
