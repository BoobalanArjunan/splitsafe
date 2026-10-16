"""Tiny synthetic audio with KNOWN relationships, so every test knows the right answer."""
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SR = 16000


def voice(seed, sec=6.0):
    """A voiced-sounding signal (moving pitch, formant-like peaks, word gating). Different seed = different recording."""
    r = np.random.RandomState(seed)
    n = int(SR * sec)
    t = np.arange(n) / SR
    f0 = 120 + 60 * r.rand()
    walk = np.cumsum(r.randn(n)) / np.sqrt(n) * 40
    ph = 2 * np.pi * np.cumsum(f0 + walk * np.sin(2 * np.pi * r.uniform(.3, 1.5) * t)) / SR
    centers = r.uniform(300, 3000, 3)
    sig = np.zeros(n)
    for k in range(1, 25):
        w = sum(np.exp(-((k * f0 - c) / (250 + 150 * r.rand())) ** 2) for c in centers)
        sig += w * np.sin(k * ph)
    gate = (np.sin(2 * np.pi * r.uniform(.8, 2) * t + r.rand() * 6) > r.uniform(-.4, 0)).astype(float)
    env = np.convolve(gate, np.ones(800) / 800, "same")
    x = 0.2 * sig / np.abs(sig).max() * env + r.randn(n) * 0.01
    return x.astype(np.float32)


def write(path: Path, audio, sr=SR):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)
    return path


def resample(x, new_sr, sr=SR):
    return signal.resample_poly(x, new_sr, sr).astype(np.float32)


def tree_hashes(root: Path):
    return {str(p.relative_to(root)): hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}
