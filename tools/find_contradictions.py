"""Outcome-based contradiction search over what the system concluded.

No per-bug instrumentation: it only reads the end results the system already
produces — identity audit samples (who was labelled where, with which method)
and the face score log — and looks for conclusions that cannot all be true:

- duplicate_person: one person on two simultaneous, non-overlapping tracks of
  the same camera (one of them is someone else)
- teleport:         one person labelled on two cameras at the same moment in
  zones neither camera shares (zones are taken from the audit records' boxes
  via the zone config)
- label_flip:       one global track carried different names over its life
- face_disagrees:   a named track whose own face attempts favour another person
  (top1 person != label in most attempts, even below the accept threshold)
- absent_person:    a name assigned while that person is known to be away
  (--absent Judit)

Every finding lists the audit images to look at; the review (skill
review-identities) confirms or dismisses it visually and proposes a fix.

Usage:
    PYTHONPATH=. python tools/find_contradictions.py local/remote/metrics \\
        --since 2026-10-17T10:00 [--config local/remote/config.jetson-live.yaml] [--absent Judit]
"""

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from tools.audit_sheet import load_samples

SIMULTANEOUS_SEC = 2.0
MIN_FACE_ATTEMPTS = 4


def _t(ts: str) -> float:
    return datetime.fromisoformat(ts).timestamp()


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _zones_of(zone_manager, camera: str, bbox) -> set[str]:
    """Zone of the box's foot point (bbox normalized -> 1x1 frame)."""
    if zone_manager is None:
        return set()
    zone = zone_manager.get_person_zone(camera, tuple(bbox), 1, 1)
    return {zone} if zone else set()


def find(samples: list[dict], face_attempts: list[dict], zone_manager=None,
         absent: tuple[str, ...] = ()) -> list[dict]:
    findings: list[dict] = []

    # duplicate_person — within one sample
    for s in samples:
        named = [t for t in s["tracks"] if t.get("person")]
        for i, a in enumerate(named):
            for b in named[i + 1:]:
                if a["person"] == b["person"] and _iou(a["bbox"], b["bbox"]) < 0.3:
                    findings.append({"kind": "duplicate_person", "ts": s["ts"], "camera": s["camera_id"],
                                     "person": a["person"], "tracks": [a.get("global_id"), b.get("global_id")],
                                     "images": [s["image"], a.get("crop"), b.get("crop")]})

    # teleport — same person, different cameras, same moment, disjoint zones
    if zone_manager is not None:
        for i, s in enumerate(samples):
            for s2 in samples[i + 1:]:
                if _t(s2["ts"]) - _t(s["ts"]) > SIMULTANEOUS_SEC:
                    break
                if s2["camera_id"] == s["camera_id"]:
                    continue
                for a in s["tracks"]:
                    for b in s2["tracks"]:
                        if not a.get("person") or a.get("person") != b.get("person"):
                            continue
                        za = _zones_of(zone_manager, s["camera_id"], a["bbox"])
                        zb = _zones_of(zone_manager, s2["camera_id"], b["bbox"])
                        if za and zb and not za & zb:
                            findings.append({"kind": "teleport", "ts": s["ts"], "person": a["person"],
                                             "where": [f"{s['camera_id']}:{sorted(za)}", f"{s2['camera_id']}:{sorted(zb)}"],
                                             "images": [s["image"], s2["image"], a.get("crop"), b.get("crop")]})

    # label_flip — one global track, several names
    names: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
    for s in samples:
        for t in s["tracks"]:
            if t.get("global_id") and t.get("person"):
                names[t["global_id"]].append((s["ts"], t["person"], {"image": s["image"], "crop": t.get("crop")}))
    for gid, seq in names.items():
        if len({p for _, p, _ in seq}) > 1:
            findings.append({"kind": "label_flip", "track": gid, "sequence": [(ts[11:19], p) for ts, p, _ in seq],
                             "images": [x["crop"] for _, _, x in seq]})

    # face_disagrees — the track's own faces vote for someone else
    label_of: dict[str, str] = {}
    evidence: dict[str, dict] = {}
    for gid, seq in names.items():
        label_of[gid] = Counter(p for _, p, _ in seq).most_common(1)[0][0]
        evidence[gid] = seq[-1][2]
    votes: dict[str, Counter] = defaultdict(Counter)
    person_names = {}
    for a in face_attempts:
        if a.get("ambiguous", a.get("num_faces", 1) > 1) or a.get("top1_person_id") is None:
            continue
        votes[a["track_id"]][int(a["top1_person_id"])] += 1
        person_names.update(a.get("_names", {}))
    for gid, label in label_of.items():
        v = votes.get(gid)
        if not v or sum(v.values()) < MIN_FACE_ATTEMPTS:
            continue
        top_pid, top_n = v.most_common(1)[0]
        top_name = person_names.get(top_pid, str(top_pid))
        if top_name != label and top_n / sum(v.values()) >= 0.6:
            findings.append({"kind": "face_disagrees", "track": gid, "label": label,
                             "faces_vote": f"{top_name} {top_n}/{sum(v.values())}",
                             "images": [evidence[gid]["image"], evidence[gid]["crop"]]})

    # absent_person
    for s in samples:
        for t in s["tracks"]:
            if t.get("person") in absent:
                findings.append({"kind": "absent_person", "ts": s["ts"], "camera": s["camera_id"],
                                 "person": t["person"], "method": t.get("method"), "track": t.get("global_id"),
                                 "images": [s["image"], t.get("crop")]})

    return findings


def load_face_attempts(metrics: Path, since: Optional[str], names: dict[int, str]) -> list[dict]:
    attempts = []
    for f in sorted((metrics / "face_scores").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since and r["ts"] < since:
                continue
            r["_names"] = names
            attempts.append(r)
    return attempts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("metrics", type=Path)
    p.add_argument("--since")
    p.add_argument("--until")
    p.add_argument("--config", type=Path, help="live config (zones) for the teleport check")
    p.add_argument("--persons", default="1:Ati,2:Judit,3:Virag,4:Rozika",
                   help="person_id:name pairs (face log stores ids)")
    p.add_argument("--absent", action="append", default=[], help="person known to be away (repeatable)")
    args = p.parse_args()

    zone_manager = None
    if args.config:
        from src.config import load_config
        from src.tracking.zone_manager import ZoneManager
        zone_manager = ZoneManager(load_config(str(args.config)).zones)

    names = {int(k): v for k, v in (pair.split(":") for pair in args.persons.split(","))}
    samples = load_samples(args.metrics, args.since, args.until, None)
    findings = find(samples, load_face_attempts(args.metrics, args.since, names), zone_manager, tuple(args.absent))
    print(f"{len(samples)} audit samples, {len(findings)} findings")
    for kind, n in Counter(f["kind"] for f in findings).most_common():
        print(f"  {kind}: {n}")
    for f in findings:
        print(json.dumps(f, ensure_ascii=False))


if __name__ == "__main__":
    main()
