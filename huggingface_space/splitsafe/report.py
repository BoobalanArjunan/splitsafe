from __future__ import annotations

import csv
import json
from pathlib import Path

from .scan import Result

NOTE = ("Note: a PASS means none of these checks found a problem. It is not a guarantee: only exact copies, "
        "same-audio copies and the speaker labels you provided are checked.")


def to_dict(res: Result, strict_speakers=False):
    return {
        "verdict": res.verdict(strict_speakers),
        "splits": res.splits,
        "min_votes": res.min_votes,
        "cross_split_duplicates": [m.__dict__ for m in res.matches],
        "shared_speakers": res.shared_speakers if res.speakers_checked else None,
        "speakers_checked": res.speakers_checked,
        "files_not_checked": [{"file": i.path, "reason": i.skipped} for i in res.skipped],
        "note": NOTE,
    }


def to_text(res: Result, strict_speakers=False, limit=15) -> str:
    v = res.verdict(strict_speakers)
    L = [f"splitsafe verdict: {v}", "Splits: " + ", ".join(f"{k} ({n} files)" for k, n in res.splits.items()), ""]
    if res.matches:
        L.append(f"LEAKAGE: {len(res.matches)} file pair(s) appear on both sides of a split "
                 f"({len(res.leaked_files)} files involved).")
        for m in res.matches[:limit]:
            L.append(f"  [{m.kind}] {m.a}  <->  {m.b}   (aligned matches: {m.votes}, overlap: {m.coverage:.0%})")
        if len(res.matches) > limit:
            L.append(f"  ... and {len(res.matches) - limit} more (see the report file)")
        L.append("")
    else:
        L.append("No duplicate audio found across splits.")
        L.append("")
    if res.speakers_checked:
        if res.shared_speakers:
            L.append(f"SPEAKERS in more than one split: {len(res.shared_speakers)}")
            for s, d in list(res.shared_speakers.items())[:limit]:
                L.append(f"  {s}: " + ", ".join(f"{k}={n}" for k, n in d.items()))
            if len(res.shared_speakers) > limit:
                L.append(f"  ... and {len(res.shared_speakers) - limit} more")
            L.append("  (For speech recognition this can be acceptable; for speaker ID or voice cloning it is usually a leak.)")
            L.append("")
        else:
            L.append("No speaker appears in more than one split.")
            L.append("")
    else:
        L.append("Speaker overlap was NOT checked (no speaker labels). Use --manifest with a speaker column or --speaker-regex.")
        L.append("")
    if res.skipped:
        L.append(f"NOT fully checked: {len(res.skipped)} file(s)")
        for i in res.skipped[:5]:
            L.append(f"  {i.path}: {i.skipped}")
        if len(res.skipped) > 5:
            L.append(f"  ... and {len(res.skipped) - 5} more")
        L.append("")
    L.append(NOTE)
    return "\n".join(L)


def write_reports(res: Result, out: Path, strict_speakers=False):
    out.mkdir(parents=True, exist_ok=True)
    (out / "splitsafe_report.json").write_text(json.dumps(to_dict(res, strict_speakers), indent=2))
    with open(out / "cross_split_duplicates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_a", "split_a", "file_b", "split_b", "kind", "aligned_matches", "overlap"])
        for m in res.matches:
            w.writerow([m.a, m.split_a, m.b, m.split_b, m.kind, m.votes, m.coverage])
    (out / "splitsafe_report.txt").write_text(to_text(res, strict_speakers, limit=10**9))
