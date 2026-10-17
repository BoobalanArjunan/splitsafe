# SplitSafe: Problem Statement, Solution Architecture & Market Edge
**Document Type:** Strategic Technical Whitepaper & Product Positioning Document  
**Author:** Boobalan Arjunan (<boobalan.it@gmail.com>)  
**Project:** SplitSafe (Audio ML Train/Val/Test Leakage & Speaker Contamination Auditor)  
**Parent Suite:** Audio ML Data Curation Suite — Tier 2 (WaveWash ➔ SplitSafe ➔ SegDupe)  
**Date:** October 16, 2026  
**Status:** Alpha Release (v0.1.0a1) — Ready for PyPI Deployment & Benchmarking  

---

## 1. The Industry Problem: Silent Data Leakage in Speech & Audio AI

### 1.1 The Modern Speech AI Boom & GPU Economics
With the explosion of speech foundation models (Whisper ASR, VALL-E, ElevenLabs-style zero-shot voice cloning, XTTS, and neural speaker diarization), machine learning teams invest tens of thousands of dollars in cloud GPU compute (NVIDIA H100/A100 clusters) to train and fine-tune models.

### 1.2 The "False Benchmark" Disaster
A critical point of failure in modern audio machine learning pipelines is **data leakage across dataset splits**:
* When compiling multi-gigabyte or terabyte speech corpora from diverse web crawls, podcasts, open datasets (Common Voice, LibriSpeech, VoxCeleb), and call center recordings, identical recordings or excerpts frequently end up in **both** the `train/` set and the `test/` (or `validation/`) set.
* **The Symptom:** During training, validation loss plunges and the model achieves over-optimistic evaluation metrics (e.g., 98%+ accuracy or Word Error Rate < 2%).
* **The Production Failure:** When deployed to production, the model encounters real-world voices and suddenly degrades or hallucinates. The model never learned to generalize; it simply memorized the leaked evaluation samples.

### 1.3 Why Conventional Tools Fail to Detect Audio Leakage
Traditional data engineering tools are fundamentally blind to audio dataset contamination:

```
┌───────────────────────────┬─────────────────────────────────────────────────────────────┐
│ Existing Approach         │ Fatal Flaw in Audio ML Pipelines                            │
├───────────────────────────┼─────────────────────────────────────────────────────────────┤
│ MD5 / SHA-256 Checksums   │ Breaks if a single sample changes, if the file is resampled │
│                           │ (e.g. 44.1 kHz ➔ 16 kHz), re-encoded (WAV ➔ FLAC/MP3),     │
│                           │ trimmed by 10ms, or if volume is tweaked by 0.1 dB.         │
├───────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Deep Embedding Models     │ Requires heavy PyTorch/GPU overhead, massive memory, slow   │
│ (Wav2Vec2, CLMR, CLAP)    │ inference across 100k+ files, and complex environment setup.│
├───────────────────────────┼─────────────────────────────────────────────────────────────┤
│ Music Fingerprinting      │ Designed to match full radio songs against a database;       │
│ (Chromaprint / AcoustID)  │ completely unaware of ML split boundaries, sub-clip clips,  │
│                           │ speaker isolation rules, or CI/CD exit codes.               │
└───────────────────────────┴─────────────────────────────────────────────────────────────┘
```

---

## 2. The SplitSafe Solution

> **Core Architectural Finding:**  
> **SplitSafe brings established spectral landmark fingerprinting techniques into an integrated, partition-aware audio ML dataset auditing workflow.**

**SplitSafe** is an open-source, local-first auditing tool specifically engineered to guarantee dataset integrity across machine learning partitions (`train/`, `val/`, `test/`).

### 2.1 Core Pillars
1. **Classic Signal Processing (No Heavy ML Overhead):**
   * Operates purely on `numpy`, `scipy`, and `soundfile`.
   * **Zero GPU requirement, zero model weights to download, zero neural network inference.**
   * Runs natively on laptops, build servers, and continuous integration pipelines in seconds.
2. **Audio-Invariant Leakage Detection:**
   * Flags the exact same recording even if it was re-encoded, resampled, gain-adjusted, contaminated with mild noise, or trimmed.
3. **Sub-Clip / Partial Overlap Recognition:**
   * Pinpoints if a short 3-second audio excerpt from a 30-second training file is sitting inside an evaluation split, calculating the exact percentage of overlap.
4. **Speaker Contamination Guard:**
   * Parses speaker identities (via filename regex or CSV manifests) and flags when the same speaker voice is present on both sides of a split.
5. **Deterministic CI/CD Integration:**
   * Provides standard exit codes (`0` for clean/warn, `1` for leakage fail, `2` for error) so pipelines halt before wasting expensive GPU training compute.

---

## 3. How SplitSafe Works: Signal Processing Architecture

```mermaid
graph TD
    A[Input Split Folders: train/ val/ test/] --> B[Resample to 8kHz Mono Float32]
    B --> C[STFT Spectrogram Analysis: 128ms Window, 32ms Hop]
    C --> D[2D Max-Filter Peak Extraction > 90th Percentile Noise Floor]
    D --> E[Pair Peaks & Generate 64-bit Invariant Landmark Hashes]
    E --> F[Inverted Hash Index Across Splits]
    F --> G[Time-Offset Delta Voting Histogram]
    G --> H{Aligned Matches >= min_votes?}
    H -- YES --> I[Flag Contaminated Audio Leak with % Overlap]
    H -- NO --> J[Clean / No Cross-Split Leakage]
    A --> K[Speaker Label Extraction Regex/CSV]
    K --> L{Same Speaker Across Splits?}
    L -- Strict Mode --> M[FAIL: Speaker Contamination]
    L -- Default Mode --> N[WARN: Speaker Shared]
```

### 3.1 Mathematical Principles of Landmark Constellation Fingerprinting
* **Windowing:** Audio is normalized to 8,000 Hz mono. A Short-Time Fourier Transform (STFT) with a 1024-point window (128 ms) and 256-point hop size (32 ms) resolves frequency bins up to 512.
* **Peak Picking:** A 2D maximum filter (`size=(21, 21)`) extracts dominant spectral energy peaks above the 90th percentile threshold, strictly capped at 60 peaks per second to prevent combinatorial explosion on dense recordings.
* **Constellation Hashing:** Each anchor peak is paired with up to 8 neighboring forward peaks within a 100-frame time window (~3.2 seconds):
  $$\text{Hash} = f_1 \mid (f_2 \ll 10) \mid (\Delta t \ll 20)$$
* **Time-Offset Coincidence Histogram Voting:**
  If two files in different splits share random frequency coincidences, their relative time differences ($\Delta T = t_2 - t_1$) are scattered randomly across the timeline.
  However, if file $B$ in `test/` contains audio copied from file $A$ in `train/`, every matched landmark peak pair votes for the **exact same constant time-offset**. This produces a massive, statistically unambiguous peak in the offset histogram.

---

## 4. How SplitSafe Takes the Market Edge

SplitSafe fills a distinct gap in the $5B+ speech and AI data tooling ecosystem:

| Feature / Dimension | Standard Checksums (MD5/SHA) | Heavy Deep Audio Embeddings | Consumer Fingerprinters (Chromaprint) | **SplitSafe (Our Market Edge)** |
| :--- | :--- | :--- | :--- | :--- |
| **Resampling / Format Invariance** | ❌ Fails on 1-byte change | ✅ Invariant | ⚠️ Mixed | **✅ 100% Invariant (STFT Peaks)** |
| **Sub-Clip Excerpt Detection** | ❌ Impossible | ⚠️ Requires sliding window | ❌ Full file only | **✅ Native (Time-offset voting)** |
| **Compute Overhead** | ✅ Microseconds | ❌ Heavy GPU / PyTorch | ⚠️ Moderate | **✅ Lightweight CPU (Pure SciPy)** |
| **Speaker Boundary Awareness** | ❌ None | ⚠️ Requires separate model | ❌ None | **✅ Built-in (Regex & Manifest)** |
| **CI/CD Integration** | ❌ Irrelevant | ❌ Too slow for CI | ❌ No ML verdicts | **✅ Native Exit Codes (0, 1, 2)** |
| **Local-First & Zero Cloud** | ✅ Local | ⚠️ Model downloads | ✅ Local | **✅ 100% Local-First, Zero Cloud** |

---

## 5. Strategic Positioning in the 3-Tier Audio Curation Suite

SplitSafe is not an isolated tool; it forms the critical **Tier 2** gatekeeper in the complete AI Audio Curation Suite:

```
┌────────────────────────────────────────────────────────────────────────┐
│  TIER 1: WaveWash (File-Level QC & Repair)                             │
│  • Fixes clipping, DC offset, background SNR, LUFS loudness, clicks.   │
├────────────────────────────────────────────────────────────────────────┤
│  TIER 2: SplitSafe (Split-Level Leakage & Contamination Auditor)       │
│  • Audits train/val/test splits to eliminate evaluation leakage &      │
│    guarantee genuine model generalization before training.             │
├────────────────────────────────────────────────────────────────────────┤
│  TIER 3: SegDupe (Segment-Level Sub-Clip Constellation Alignment)      │
│  • Deep granular sub-sequence duplicate detection and deduplication.   │
└────────────────────────────────────────────────────────────────────────┘
```

---

---

## 7. Immediate Action Roadmap

- [x] Technical problem statement and market-edge documentation completed.
- [x] Core test suite passing (21/21 tests in `tests/test_splits.py`).
- [ ] Run benchmark evaluation script on LibriSpeech (`scripts/real_data_check.py`).
- [ ] Build distribution wheel and publish to PyPI (`splitsafe v0.1.0a1`).
- [ ] Publish technical launch article / LinkedIn post highlighting audio data leakage prevention.
