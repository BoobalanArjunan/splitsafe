"""Find leakage between splits: cross-split duplicates (exact + near) and shared speakers."""
from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .fingerprint import DT_MASK, DT_SHIFT, fingerprint, load_mono

AUDIO_EXT = {".wav", ".flac", ".ogg", ".opus", ".mp3", ".m4a", ".aiff", ".aif"}
DEFAULT_MIN_VOTES = 30      # aligned hash matches needed to call two files "the same audio"


@dataclass
class Item:
    path: str          # path as shown to the user
    split: str
    speaker: str | None = None
    sha1: str = ""
    n_hashes: int = 0
    seconds: float = 0.0
    skipped: str = ""  # reason it could not be fingerprinted
    abs: Path | None = None


@dataclass
class Match:
    a: str
    b: str
    split_a: str
    split_b: str
    kind: str          # "exact copy" | "same audio (re-encoded / trimmed / re-leveled)"
    votes: int
    coverage: float    # aligned matches / hashes in the smaller file (0..1)
    abs_a: Path | None = None
    abs_b: Path | None = None


@dataclass
class Result:
    splits: dict[str, int]
    matches: list[Match] = field(default_factory=list)
    shared_speakers: dict[str, dict[str, int]] = field(default_factory=dict)   # speaker -> split -> files
    skipped: list[Item] = field(default_factory=list)
    speakers_checked: bool = False
    min_votes: int = DEFAULT_MIN_VOTES

    @property
    def leaked_files(self) -> set[str]:
        s = set()
        for m in self.matches:
            s.add(m.a)
            s.add(m.b)
        return s

    def verdict(self, strict_speakers: bool = False) -> str:
        if self.matches:
            return "FAIL"
        if self.shared_speakers:
            return "FAIL" if strict_speakers else "WARN"
        return "PASS"


def _list_audio(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXT)


def _sha1(path: Path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def items_from_folders(folders: list[Path], names: list[str] | None = None) -> list[Item]:
    names = names or [f.name for f in folders]
    if len(set(names)) != len(names):
        raise ValueError(f"split names must be different, got {names}. Use --names a,b,c")
    out = []
    for folder, name in zip(folders, names):
        if not folder.is_dir():
            raise ValueError(f"not a folder: {folder}")
        for p in _list_audio(folder):
            out.append(_mk(p, name, f"{name}/{p.relative_to(folder).as_posix()}"))
    return out


def _mk(abs_path: Path, split: str, shown: str) -> Item:
    return Item(path=shown, split=split, abs=abs_path)


def items_from_manifest(manifest: Path, base: Path | None = None) -> list[Item]:
    """CSV with columns: file, split and optionally speaker. file is relative to --base (default: the CSV's folder)."""
    base = base or manifest.parent
    out = []
    with open(manifest, newline="", encoding="utf-8-sig") as f:
        rd = csv.DictReader(f)
        cols = {c.strip().lower(): c for c in (rd.fieldnames or [])}
        if "file" not in cols or "split" not in cols:
            raise ValueError("manifest needs columns: file, split (and optionally speaker)")
        for row in rd:
            rel = row[cols["file"]].strip()
            p = (base / rel) if not Path(rel).is_absolute() else Path(rel)
            it = _mk(p, row[cols["split"]].strip(), rel)
            if "speaker" in cols and row[cols["speaker"]].strip():
                it.speaker = row[cols["speaker"]].strip()
            out.append(it)
    return out


def apply_speaker_regex(items: list[Item], pattern: str):
    rx = re.compile(pattern)
    if rx.groups < 1:
        raise ValueError("--speaker-regex needs one (capture group), e.g. '^[^/]+/([^/_-]+)'")
    for it in items:
        m = rx.search(it.path)
        if m:
            it.speaker = m.group(1)


def _build_index(fps):
    H = np.concatenate([h for h, _, _ in fps]) if fps else np.empty(0, np.int64)
    T = np.concatenate([t for _, t, _ in fps]) if fps else np.empty(0, np.int32)
    F = np.concatenate([np.full(len(h), i, np.int32) for i, (h, _, _) in enumerate(fps)]) if fps else np.empty(0, np.int32)
    order = np.argsort(H, kind="stable")
    return H[order], T[order], F[order]


def _query(H, T, F, qh, qt, self_id, split_id):
    """Return {other_file_id: best_votes}: hash matches that agree on ONE time offset.

    A copy that was trimmed or re-encoded can shift peaks by one frame, so each query hash
    is also looked up with its time gap +-1, and neighbouring offsets are added together."""
    dts = (qh >> DT_SHIFT) & DT_MASK
    qs, ts = [], []
    for d in (0, -1, 1):
        ok = (dts + d) >= 1
        qs.append(qh[ok] + (d << DT_SHIFT))
        ts.append(qt[ok])
    qh2, qt2 = np.concatenate(qs), np.concatenate(ts)
    left = np.searchsorted(H, qh2, "left")
    right = np.searchsorted(H, qh2, "right")
    cnt = right - left
    # very common hashes carry little information and can blow up memory: skip them
    ok = (cnt > 0) & (cnt <= 2000)
    if not ok.any():
        return {}
    left, cnt, qt_ok = left[ok], cnt[ok], qt2[ok]
    total = int(cnt.sum())
    start = np.repeat(left - np.concatenate([[0], np.cumsum(cnt)[:-1]]), cnt)
    idx = start + np.arange(total)
    fid = F[idx]
    dt = T[idx].astype(np.int64) - np.repeat(qt_ok, cnt).astype(np.int64)
    keep = (fid != self_id) & (split_id[fid] != split_id[self_id])
    if not keep.any():
        return {}
    fid, dt = fid[keep].astype(np.int64), dt[keep]
    key = fid * 1_000_000 + (dt + 500_000)
    uk, c = np.unique(key, return_counts=True)
    counts = dict(zip(uk.tolist(), c.tolist()))
    best = {}
    for k, v in counts.items():
        v2 = v + counts.get(k + 1, 0)          # offset and its neighbour
        f = k // 1_000_000
        if v2 > best.get(f, 0):
            best[f] = v2
    return best


def check_splits(items: list[Item], min_votes: int = DEFAULT_MIN_VOTES, progress=None) -> Result:
    split_names = sorted({i.split for i in items})
    if len(split_names) < 2:
        raise ValueError("need at least two splits to compare")
    res = Result(splits={s: sum(1 for i in items if i.split == s) for s in split_names},
                 min_votes=min_votes, speakers_checked=any(i.speaker for i in items))

    fps = []   # aligned with `usable`
    usable: list[Item] = []
    for n, it in enumerate(items):
        if progress:
            progress(n + 1, len(items))
        p = it.abs
        if not p.is_file():
            it.skipped = "file not found"
            res.skipped.append(it)
            continue
        try:
            it.sha1 = _sha1(p)
            x, _ = load_mono(p)
        except Exception as e:  # unreadable / corrupt: say so, do not pretend it was checked
            it.skipped = f"could not read audio ({type(e).__name__})"
            res.skipped.append(it)
            continue
        it.seconds = len(x) / 8000
        h, t = fingerprint(x)
        it.n_hashes = len(h)
        if len(h) == 0:
            it.skipped = "too short or silent to fingerprint (only exact byte copies are checked)"
            res.skipped.append(it)
            # still keep for exact-copy detection below
        else:
            usable.append(it)
            fps.append((h, t, it))

    # --- exact byte copies across splits (works for every readable file, even short ones)
    by_sha: dict[str, list[Item]] = {}
    for it in items:
        if it.sha1:
            by_sha.setdefault(it.sha1, []).append(it)
    seen = set()
    for group in by_sha.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if a.split != b.split:
                    key = tuple(sorted((a.path, b.path)))
                    seen.add(key)
                    first_it, second_it = (a, b) if a.path == key[0] else (b, a)
                    res.matches.append(Match(key[0], key[1],
                                             first_it.split, second_it.split,
                                             "exact copy", max(a.n_hashes, b.n_hashes), 1.0,
                                             first_it.abs, second_it.abs))

    # --- same audio, different bytes (re-encoded, resampled, trimmed, louder/quieter)
    if fps:
        H, T, F = _build_index([(h, t, i) for h, t, i in fps])
        sid_map = {s: k for k, s in enumerate(split_names)}
        split_id = np.array([sid_map[i.split] for _, _, i in fps], np.int32)
        pair_best: dict[tuple[int, int], int] = {}
        for qi, (qh, qt, it) in enumerate(fps):
            for fj, v in _query(H, T, F, qh, qt, qi, split_id).items():
                if v >= min_votes:
                    k = (min(qi, fj), max(qi, fj))
                    pair_best[k] = max(pair_best.get(k, 0), v)
        for (a, b), v in sorted(pair_best.items()):
            ia, ib = fps[a][2], fps[b][2]
            key = tuple(sorted((ia.path, ib.path)))
            if key in seen:
                continue
            smaller = max(1, min(ia.n_hashes, ib.n_hashes))
            first, second = sorted((ia, ib), key=lambda z: z.path)
            res.matches.append(Match(first.path, second.path, first.split, second.split,
                                     "same audio (re-encoded / trimmed / re-leveled)", v, round(min(1.0, v / smaller), 3),
                                     first.abs, second.abs))

    # --- speakers present in more than one split
    spk: dict[str, dict[str, int]] = {}
    for it in items:
        if it.speaker:
            d = spk.setdefault(it.speaker, {})
            d[it.split] = d.get(it.split, 0) + 1
    res.shared_speakers = {s: d for s, d in sorted(spk.items()) if len(d) > 1}
    res.matches.sort(key=lambda m: (-m.votes, m.a))
    return res
