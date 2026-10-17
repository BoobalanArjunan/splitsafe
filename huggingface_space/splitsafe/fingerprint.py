"""Landmark audio fingerprint (spectral peak pairs), classic signal processing, no ML.

Idea: find loud spectral peaks, pair each with a few peaks that come shortly after,
and hash (freq1, freq2, time gap). Two files that share audio share many hashes at
one *consistent time offset*. That survives re-encoding, resampling, gain changes,
trimming and mild noise, and it also finds a short clip cut out of a longer file.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import maximum_filter

FP_SR = 8000
NPERSEG = 1024       # 128 ms window at 8 kHz
HOP = 256            # 32 ms per frame
FAN_OUT = 8
MAX_DT = 100         # frames (~3.2 s)
PEAKS_PER_SEC = 60
F_BITS = 10          # frequency bins go up to NPERSEG/2 = 512, needs 10 bits
DT_SHIFT = 2 * F_BITS
DT_MASK = 0x7F
MIN_SECONDS = 1.0    # shorter than this is not fingerprinted (reported as skipped)


def load_mono(path, target_sr=FP_SR):
    """Read any soundfile-readable file -> mono float32 at target_sr. Returns (audio, original_sr)."""
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != target_sr:
        g = np.gcd(int(sr), int(target_sr))
        x = signal.resample_poly(x, target_sr // g, sr // g).astype(np.float32)
    return x, sr


def fingerprint(x: np.ndarray):
    """-> (hashes int64[], frames int32[]). Empty arrays if the audio is too short or silent."""
    empty = (np.empty(0, np.int64), np.empty(0, np.int32))
    if len(x) < FP_SR * MIN_SECONDS or not np.any(np.abs(x) > 1e-6):
        return empty
    _, _, Z = signal.stft(x, fs=FP_SR, nperseg=NPERSEG, noverlap=NPERSEG - HOP, boundary=None, padded=False)
    L = 20 * np.log10(np.abs(Z) + 1e-10)
    local_max = maximum_filter(L, size=(21, 21), mode="nearest") == L
    floor = np.percentile(L, 90)
    ys, xs = np.nonzero(local_max & (L > floor))
    if len(xs) < 2:
        return empty
    amp = L[ys, xs]
    # keep the strongest peaks only, bounded per second, so dense music does not explode
    keep_n = max(2, int(PEAKS_PER_SEC * len(x) / FP_SR))
    if len(xs) > keep_n:
        top = np.argsort(amp)[-keep_n:]
        ys, xs = ys[top], xs[top]
    order = np.lexsort((ys, xs))
    ys, xs = ys[order].astype(np.int64), xs[order].astype(np.int64)
    hs, ts = [], []
    n = len(xs)
    for k in range(1, FAN_OUT + 1):
        if n - k <= 0:
            break
        dt = xs[k:] - xs[:-k]
        ok = (dt >= 1) & (dt <= MAX_DT)
        if not ok.any():
            continue
        f1, f2 = ys[:-k][ok], ys[k:][ok]
        hs.append(f1 | (f2 << F_BITS) | (dt[ok] << DT_SHIFT))
        ts.append(xs[:-k][ok])
    if not hs:
        return empty
    h = np.concatenate(hs)
    t = np.concatenate(ts).astype(np.int32)
    # drop exact repeats of the same (hash, time) produced by overlapping fan-outs
    key = np.unique(np.stack([h, t.astype(np.int64)], 1), axis=0)
    return key[:, 0].astype(np.int64), key[:, 1].astype(np.int32)
