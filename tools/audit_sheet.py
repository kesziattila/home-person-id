"""Build contact sheets from identity audit samples for visual spot-check review.

Input: `<metrics>/audit/*.jsonl` + `<metrics>/audit_frames/` (scripts/remote/fetch_audit.sh).
Output (in --out):
- `label_<name>.jpg`  — crops of every track the system labelled <name>, up to
  --per-track crops per track spread over its lifetime (catches mid-track
  identity switches), reference gallery faces on the first row (--refs)
- `label_unknown.jpg` — crops of unlabelled tracks (missed identifications,
  false-positive detections)
- `changes_<n>.jpg`   — annotated frames of label-change samples (context:
  who was where, overlapping boxes, zones)
- `index.txt`         — cell -> (time, camera, local/global id, label, method,
  score, crop path), to correlate with face_scores / events / app.log

Usage:
    PYTHONPATH=. python tools/audit_sheet.py local/remote/metrics \\
        --since 2026-10-17T10:00 --refs local/remote/refs/data/faces --out data/debug/audit
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

CELL_W, CELL_H = 150, 260
FRAME_W = 640


def load_samples(metrics: Path, since: Optional[str], until: Optional[str], camera: Optional[str]) -> list[dict]:
    samples = []
    for f in sorted((metrics / "audit").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue
            if since and s["ts"] < since or until and s["ts"] >= until:
                continue
            if camera and s["camera_id"] != camera:
                continue
            samples.append(s)
    return sorted(samples, key=lambda s: s["ts"])


def _cell(img: Optional[np.ndarray], caption: list[str], w: int = CELL_W, h: int = CELL_H) -> np.ndarray:
    cell = np.zeros((h, w, 3), dtype=np.uint8)
    if img is not None and img.size:
        scale = min(w / img.shape[1], (h - 30) / img.shape[0])
        r = cv2.resize(img, (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))))
        y0, x0 = 30 + (h - 30 - r.shape[0]) // 2, (w - r.shape[1]) // 2
        cell[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    for i, text in enumerate(caption[:2]):
        cv2.putText(cell, text, (3, 12 + 13 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1)
    return cell


def _grid(cells: list[np.ndarray], cols: int) -> np.ndarray:
    blank = np.zeros_like(cells[0])
    cells = cells + [blank] * (-len(cells) % cols)
    return np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)])


def _spread(items: list, k: int) -> list:
    if len(items) <= k:
        return items
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in sorted(set(idx))]


def build(metrics: Path, samples: list[dict], out: Path, refs: Optional[Path],
          per_track: int = 3, cols: int = 10) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    index: list[str] = []

    # Track crops grouped by (label, track), spread over the track's lifetime
    by_track: dict[tuple, list[tuple[dict, dict]]] = defaultdict(list)
    for s in samples:
        for t in s["tracks"]:
            key = (t.get("person") or "unknown", t.get("global_id") or f"{s['camera_id']}#{t['local_id']}")
            by_track[key].append((s, t))
    by_label: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for (label, _), entries in sorted(by_track.items(), key=lambda kv: kv[1][0][0]["ts"]):
        by_label[label].extend(_spread(entries, per_track))

    for label, entries in sorted(by_label.items()):
        cells = []
        ref_dir = refs / label if refs else None
        if refs:  # persons are stored as person_<id>; caller may symlink names
            ref_paths = sorted(ref_dir.glob("*.jpg"))[-cols:] if ref_dir.is_dir() else []
            cells += [_cell(cv2.imread(str(p)), ["REF " + label]) for p in ref_paths]
            cells += [_cell(None, [])] * (-len(cells) % cols)
        for n, (s, t) in enumerate(entries):
            crop = cv2.imread(str(metrics / t["crop"])) if t.get("crop") else None
            tag = f"{label}#{n}"
            cells.append(_cell(crop, [
                f"{n} {s['ts'][11:19]} {s['camera_id'][:6]}",
                f"{(t.get('method') or '-')[:9]} {t.get('confidence', 0):.2f}",
            ]))
            index.append(f"{tag}\t{s['ts']}\t{s['camera_id']}\tlocal={t['local_id']}\t"
                         f"global={t.get('global_id')}\t{t.get('method')}\t{t.get('confidence')}\t"
                         f"face={t.get('face_info')}\t{t.get('crop')}")
        if cells:
            cv2.imwrite(str(out / f"label_{label}.jpg"), _grid(cells, cols))

    # Context frames of label changes
    changes = [s for s in samples if s.get("trigger") == "label_change"]
    for page in range(0, len(changes), 6):
        cells = []
        for s in changes[page:page + 6]:
            frame = cv2.imread(str(metrics / s["image"]))
            names = ", ".join(f"t{t['local_id']}:{t.get('person') or '?'}" for t in s["tracks"] if t.get("changed"))
            cells.append(_cell(frame, [f"{s['ts'][11:19]} {s['camera_id']}", names], w=FRAME_W, h=400))
            index.append(f"change{page // 6}\t{s['ts']}\t{s['camera_id']}\t{s['image']}\t{names}")
        cv2.imwrite(str(out / f"changes_{page // 6}.jpg"), _grid(cells, 2))

    (out / "index.txt").write_text("\n".join(index) + "\n")
    return index


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("metrics", type=Path, help="local metrics dir (contains audit/ and audit_frames/)")
    p.add_argument("--since", help="ISO UTC timestamp prefix, e.g. 2026-10-17T10:00")
    p.add_argument("--until")
    p.add_argument("--camera")
    p.add_argument("--refs", type=Path, help="dir with <person name>/*.jpg reference faces")
    p.add_argument("--per-track", type=int, default=3)
    p.add_argument("--out", type=Path, default=Path("data/debug/audit"))
    args = p.parse_args()

    samples = load_samples(args.metrics, args.since, args.until, args.camera)
    index = build(args.metrics, samples, args.out, args.refs, args.per_track)
    print(f"{len(samples)} samples -> {args.out} ({len(index)} index rows)")
    for f in sorted(args.out.glob("*.jpg")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
