"""Real-data check for splitsafe using LibriSpeech (or any two speaker-disjoint folders).

Two questions, answered with numbers:
  1. FALSE ALARMS: LibriSpeech dev-clean and test-clean are officially separate, so a good checker
     should find (almost) nothing between them. Anything it reports is a candidate false alarm to listen to.
  2. MISSED LEAKS: we plant copies of real dev-clean files (resampled, quieter, trimmed, noisy, clipped, exact)
     into the test side and count how many splitsafe finds. Your data is never modified.

Usage:
  python scripts/real_data_check.py --a ~/LibriSpeech/dev-clean --b ~/LibriSpeech/test-clean --plant 60 --out real_check
"""
import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from splitsafe.scan import AUDIO_EXT, Item, _mk, apply_speaker_regex, check_splits, items_from_folders

KINDS = ["exact copy", "resampled 11k", "quieter", "trimmed", "noisy ~20dB", "3 s clip"]


def make_variant(src: Path, kind: str, dst_dir: Path, tag: str, rng: random.Random) -> Path:
    x, sr = sf.read(str(src), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if kind == "exact copy":
        out = dst_dir / f"{tag}{src.suffix}"
        shutil.copyfile(src, out)
        return out
    if kind == "resampled 11k":
        y = signal.resample_poly(x, 11025, sr).astype(np.float32); out = dst_dir / f"{tag}.wav"; sf.write(str(out), y, 11025)
    elif kind == "quieter":
        out = dst_dir / f"{tag}.wav"; sf.write(str(out), (x * 0.25).astype(np.float32), sr)
    elif kind == "trimmed":
        a = int(len(x) * 0.2); out = dst_dir / f"{tag}.wav"; sf.write(str(out), x[a:], sr)
    elif kind == "noisy ~20dB":
        rms = float(np.sqrt(np.mean(x ** 2)) + 1e-9)
        n = np.random.RandomState(rng.randrange(10**6)).randn(len(x)) * rms * 0.1
        out = dst_dir / f"{tag}.wav"; sf.write(str(out), (x + n).astype(np.float32), sr)
    elif kind == "3 s clip":
        if len(x) < 5 * sr:
            raise ValueError("too short")
        s = rng.randrange(0, len(x) - 3 * sr); out = dst_dir / f"{tag}.wav"; sf.write(str(out), x[s:s + 3 * sr], sr)
    else:
        raise ValueError(kind)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="first split folder, e.g. LibriSpeech/dev-clean")
    ap.add_argument("--b", required=True, help="second split folder, e.g. LibriSpeech/test-clean")
    ap.add_argument("--plant", type=int, default=60, help="how many planted copies (default 60, spread over 6 kinds)")
    ap.add_argument("--speaker-regex", default=r"^[^/]+/(\d+)/", help="default fits LibriSpeech: <split>/<speaker>/<chapter>/file")
    ap.add_argument("--min-votes", type=int, default=30)
    ap.add_argument("--max-files", type=int, default=0, help="use only the first N files of each split (quick trial)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="real_check")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    out = Path(a.out); planted_dir = out / "planted"
    shutil.rmtree(planted_dir, ignore_errors=True); planted_dir.mkdir(parents=True)

    A, B = Path(a.a).expanduser(), Path(a.b).expanduser()
    itemsA = items_from_folders([A], ["A"])
    itemsB = items_from_folders([B], ["B"])
    if a.max_files:
        rng.shuffle(itemsA); rng.shuffle(itemsB)
        itemsA, itemsB = itemsA[:a.max_files], itemsB[:a.max_files]
    base_items = itemsA + itemsB
    apply_speaker_regex(base_items, a.speaker_regex)
    print(f"{len(itemsA)} files in A, {len(itemsB)} files in B")

    # ---- 1. false alarms
    t0 = time.time()
    base = check_splits(base_items, a.min_votes)
    t_base = time.time() - t0
    print(f"[1] baseline done in {t_base:.0f}s: cross-split pairs found = {len(base.matches)}, "
          f"shared speakers = {len(base.shared_speakers)}, files not checked = {len(base.skipped)}")
    base_votes = sorted((m.votes for m in base.matches), reverse=True)

    # ---- 2. planted leaks
    sources = rng.sample(itemsA, min(a.plant, len(itemsA)))
    planted = []   # (kind, source path, planted item)
    for i, src in enumerate(sources):
        kind = KINDS[i % len(KINDS)]
        try:
            p = make_variant(src.abs, kind, planted_dir, f"p{i:03d}", rng)
        except ValueError:
            continue
        planted.append((kind, src.path, _mk(p, "B", f"planted/{p.name}")))
    # a fresh pass: everything in A and B, plus planted files on the B side
    items2 = [Item(path=i.path, split=i.split, speaker=i.speaker, abs=i.abs) for i in base_items] + [it for _, _, it in planted]
    t0 = time.time()
    res = check_splits(items2, a.min_votes)
    t2 = time.time() - t0
    found = {}
    votes_by_kind = {k: [] for k in KINDS}
    pair_votes = {}
    for m in res.matches:
        pair_votes[frozenset((m.a, m.b))] = m.votes
    for kind, srcp, it in planted:
        v = pair_votes.get(frozenset((srcp, it.path)))
        found.setdefault(kind, []).append(v is not None)
        votes_by_kind[kind].append(v or 0)

    lines = ["# splitsafe real-data check", "",
             f"Folders: A = `{A}`, B = `{B}`. Files: {len(itemsA)} / {len(itemsB)}. min-votes = {a.min_votes}. seed = {a.seed}.", "",
             "## 1. False alarms (A and B are officially separate)", "",
             f"- Cross-split pairs reported: **{len(base.matches)}**",
             f"- Shared speakers: **{len(base.shared_speakers)}**",
             f"- Files not fully checked: {len(base.skipped)}",
             f"- Highest vote counts among reported pairs: {base_votes[:10]}",
             f"- Time: {t_base:.0f}s", ""]
    if base.matches:
        lines += ["Pairs to LISTEN to (are they really the same audio?):", ""]
        for m in base.matches[:20]:
            lines.append(f"- `{m.a}` <-> `{m.b}` votes={m.votes} overlap={m.coverage:.0%}")
        lines.append("")
    lines += ["## 2. Planted leaks (real files, changed, put on the other side)", "",
              "| Kind | planted | found | missed | min votes | median votes |", "|---|---|---|---|---|---|"]
    tot_p = tot_f = 0
    for k in KINDS:
        f = found.get(k, [])
        if not f:
            continue
        tot_p += len(f); tot_f += sum(f)
        lines.append(f"| {k} | {len(f)} | {sum(f)} | {len(f) - sum(f)} | {min(votes_by_kind[k])} | {int(np.median(votes_by_kind[k]))} |")
    lines += ["", f"**Total: {tot_f} of {tot_p} planted leaks found** ({tot_f / max(1, tot_p):.0%}). Time: {t2:.0f}s.", "",
              "If a kind has misses, lower `--min-votes` and run again, but check the false-alarm list in part 1 does not grow.", ""]
    (out / "results.md").write_text("\n".join(lines))
    (out / "results.json").write_text(json.dumps({
        "baseline_pairs": len(base.matches), "baseline_top_votes": base_votes[:20], "shared_speakers": len(base.shared_speakers),
        "planted": tot_p, "found": tot_f, "by_kind": {k: {"planted": len(found.get(k, [])), "found": sum(found.get(k, [])),
                                                         "votes": votes_by_kind[k]} for k in KINDS}}, indent=2))
    print("\n".join(lines))
    print(f"\nSaved: {out / 'results.md'}")


if __name__ == "__main__":
    main()
