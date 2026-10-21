"""Replay face-match policies against the live face score log.

Reads `<metrics.path>/face_scores/*.jsonl` (written by FaceScoreLogger, pulled
with `scripts/remote/fetch_metrics.sh`) and evaluates every
(similarity_threshold, min_margin) pair on a grid.

Pseudo-labels: a track whose attempts include an "anchor" (top1 >= 0.6 and
margin >= 0.15, all anchors agreeing on one person) is labelled with that
person. Its NON-anchor attempts are the hard cases a looser policy would
have to decide; they give a live precision/recall estimate without manual
labelling. Caveat: a ByteTrack ID switch inside a track contaminates its
label; that counts correct recognitions as errors, so the precision estimate
is conservative.

Track-level emulation mirrors IdentityLinker: a track is identified once two
accepted attempts in a row (rejections in between do not reset) agree on the
same person, or at once by one strong attempt (--strong-threshold/--strong-margin,
face_recognition.strong_*). Logged attempts are rate-limited, so real coverage is higher.

Usage:
    PYTHONPATH=. python tools/analyze_face_scores.py local/remote/metrics/face_scores \
        [--since 2026-10-16] [--min-precision 0.99] [--json out.json]
"""

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

ANCHOR_THRESHOLD = 0.6
ANCHOR_MARGIN = 0.15
THRESHOLDS = [round(0.30 + 0.02 * i, 2) for i in range(16)]  # 0.30 .. 0.60
MARGINS = [0.0, 0.05, 0.10, 0.15, 0.20]
MIN_LABELED_ACCEPTS = 100  # below this a precision estimate is not trusted


@dataclass
class Attempt:
    ts: str
    track_id: str
    camera_id: Optional[str]
    top1_person_id: int
    top1: float
    top2: float

    @property
    def margin(self) -> float:
        return self.top1 - self.top2

    def accepted(self, threshold: float, min_margin: float) -> bool:
        return self.top1 >= threshold and self.margin >= min_margin

    @property
    def is_anchor(self) -> bool:
        return self.accepted(ANCHOR_THRESHOLD, ANCHOR_MARGIN)


@dataclass
class PolicyResult:
    threshold: float
    min_margin: float
    labeled_attempts: int  # non-anchor attempts of labelled tracks
    labeled_accepted: int
    labeled_correct: int
    precision: Optional[float]  # labeled_correct / labeled_accepted
    precision_lower95: Optional[float]  # Wilson lower bound
    recall: float  # labeled_correct / labeled_attempts
    tracks_identified: int  # all tracks with >=1 attempt, emulated linker
    tracks_total: int
    labeled_tracks_wrong: int  # labelled tracks the policy would confirm as someone else

    @property
    def track_coverage(self) -> float:
        return self.tracks_identified / self.tracks_total if self.tracks_total else 0.0


@dataclass
class Report:
    attempts: int
    tracks: int
    labeled_tracks: int
    conflicted_tracks: int
    current_policy: Optional[PolicyResult]
    recommended: Optional[PolicyResult]
    reason: str
    impostor_p99: Optional[float]
    genuine_p50: Optional[float]
    grid: list[PolicyResult] = field(default_factory=list)


def load_attempts(path: Path, since: Optional[str] = None) -> list[Attempt]:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    attempts = []
    for f in files:
        if since and f.stem < since:
            continue
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # partially written last line
            if r.get("top1_person_id") is None:
                continue
            if r.get("ambiguous", r.get("num_faces", 1) > 1):
                continue  # the scored face may belong to another person
            attempts.append(Attempt(
                ts=r["ts"], track_id=r["track_id"], camera_id=r.get("camera_id"),
                top1_person_id=int(r["top1_person_id"]), top1=r["top1"], top2=r["top2"],
            ))
    attempts.sort(key=lambda a: a.ts)
    return attempts


def label_tracks(by_track: dict[str, list[Attempt]]) -> tuple[dict[str, int], int]:
    """Track -> person from agreeing anchors; returns (labels, conflicted count)."""
    labels, conflicted = {}, 0
    for track_id, attempts in by_track.items():
        persons = {a.top1_person_id for a in attempts if a.is_anchor}
        if len(persons) == 1:
            labels[track_id] = persons.pop()
        elif len(persons) > 1:
            conflicted += 1
    return labels, conflicted


def emulate_track(
    attempts: list[Attempt], threshold: float, min_margin: float,
    strong: Optional[tuple[float, float]] = None,
) -> Optional[int]:
    """Person the linker would confirm (2 accepted in a row, same person, or one strong), else None."""
    candidate = None
    for a in attempts:
        if not a.accepted(threshold, min_margin):
            continue
        if a.top1_person_id == candidate or (strong and a.accepted(*strong)):
            return a.top1_person_id
        candidate = a.top1_person_id
    return None


def wilson_lower(successes: int, n: int, z: float = 1.96) -> Optional[float]:
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - spread) / denom


def evaluate(
    by_track: dict[str, list[Attempt]], labels: dict[str, int], threshold: float, min_margin: float,
    strong: Optional[tuple[float, float]] = None,
) -> PolicyResult:
    n = accepted = correct = identified = wrong_tracks = 0
    for track_id, attempts in by_track.items():
        confirmed = emulate_track(attempts, threshold, min_margin, strong)
        if confirmed is not None:
            identified += 1
        label = labels.get(track_id)
        if label is None:
            continue
        if confirmed is not None and confirmed != label:
            wrong_tracks += 1
        for a in attempts:
            if a.is_anchor:
                continue
            n += 1
            if a.accepted(threshold, min_margin):
                accepted += 1
                correct += a.top1_person_id == label
    return PolicyResult(
        threshold=threshold, min_margin=min_margin,
        labeled_attempts=n, labeled_accepted=accepted, labeled_correct=correct,
        precision=correct / accepted if accepted else None,
        precision_lower95=wilson_lower(correct, accepted),
        recall=correct / n if n else 0.0,
        tracks_identified=identified, tracks_total=len(by_track),
        labeled_tracks_wrong=wrong_tracks,
    )


def _percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def analyze(
    attempts: list[Attempt],
    current: tuple[float, float] = (0.6, 0.0),
    min_precision: float = 0.99,
    strong: Optional[tuple[float, float]] = None,
) -> Report:
    by_track: dict[str, list[Attempt]] = defaultdict(list)
    for a in attempts:
        by_track[a.track_id].append(a)
    labels, conflicted = label_tracks(by_track)

    grid = [evaluate(by_track, labels, t, m, strong) for t in THRESHOLDS for m in MARGINS]
    current_result = evaluate(by_track, labels, *current, strong)

    # Live genuine / impostor evidence from labelled non-anchor attempts
    genuine, impostor = [], []
    for track_id, label in labels.items():
        for a in by_track[track_id]:
            if a.is_anchor:
                continue
            if a.top1_person_id == label:
                genuine.append(a.top1)
                impostor.append(a.top2)
            else:
                impostor.append(a.top1)

    safe = [
        r for r in grid
        if r.labeled_accepted >= MIN_LABELED_ACCEPTS
        and r.precision_lower95 is not None and r.precision_lower95 >= min_precision
        and r.labeled_tracks_wrong == 0
    ]
    if safe:
        best = max(safe, key=lambda r: (r.track_coverage, r.threshold, r.min_margin))
        reason = (
            f"max track coverage with precision lower95 >= {min_precision}, "
            f">= {MIN_LABELED_ACCEPTS} labelled accepts, no wrongly confirmed labelled track"
        )
    else:
        best = None
        reason = (
            f"insufficient evidence: no policy has >= {MIN_LABELED_ACCEPTS} labelled accepts "
            f"with precision lower95 >= {min_precision} ({len(labels)} labelled tracks so far)"
        )

    return Report(
        attempts=len(attempts), tracks=len(by_track), labeled_tracks=len(labels),
        conflicted_tracks=conflicted, current_policy=current_result,
        recommended=best, reason=reason,
        impostor_p99=_percentile(impostor, 0.99), genuine_p50=_percentile(genuine, 0.5),
        grid=grid,
    )


def _fmt(r: PolicyResult) -> str:
    prec = f"{r.precision:.3f}" if r.precision is not None else "  -  "
    low = f"{r.precision_lower95:.3f}" if r.precision_lower95 is not None else "  -  "
    return (
        f"t={r.threshold:.2f} m={r.min_margin:.2f}  prec={prec} (lo95 {low}, "
        f"n={r.labeled_accepted})  recall={r.recall:.3f}  "
        f"tracks={r.tracks_identified}/{r.tracks_total} ({r.track_coverage:.1%})  "
        f"wrong_tracks={r.labeled_tracks_wrong}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("path", type=Path, help="face_scores dir or a single .jsonl file")
    parser.add_argument("--since", help="YYYY-MM-DD, skip older daily files")
    parser.add_argument("--current-threshold", type=float, default=0.6)
    parser.add_argument("--current-margin", type=float, default=0.0)
    parser.add_argument("--strong-threshold", type=float, default=0.0, help="0 = off")
    parser.add_argument("--strong-margin", type=float, default=0.15)
    parser.add_argument("--min-precision", type=float, default=0.99)
    parser.add_argument("--json", type=Path, help="write the full report as JSON")
    args = parser.parse_args()

    report = analyze(
        load_attempts(args.path, args.since),
        current=(args.current_threshold, args.current_margin),
        min_precision=args.min_precision,
        strong=(args.strong_threshold, args.strong_margin) if args.strong_threshold > 0 else None,
    )
    print(f"attempts={report.attempts} tracks={report.tracks} "
          f"labelled={report.labeled_tracks} conflicted={report.conflicted_tracks}")
    print(f"live genuine p50={report.genuine_p50}  impostor p99={report.impostor_p99}")
    print("current:     " + _fmt(report.current_policy))
    print("recommended: " + (_fmt(report.recommended) if report.recommended else "none"))
    print(f"reason: {report.reason}")
    for m in MARGINS:
        row = [r for r in report.grid if r.min_margin == m and r.threshold in (0.36, 0.42, 0.48, 0.54, 0.6)]
        print(f"-- margin {m:.2f}")
        for r in row:
            print("   " + _fmt(r))
    if args.json:
        args.json.write_text(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
