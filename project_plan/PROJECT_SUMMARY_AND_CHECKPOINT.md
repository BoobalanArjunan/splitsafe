# SplitSafe: Master Project Plan, Architecture & Checkpoint Log
**Author:** Boobalan Arjunan (<boobalan.it@gmail.com>)  
**Project:** SplitSafe (Audio Train/Val/Test Leakage & Speaker Contamination Detection for Speech & Audio ML)  
**Date Created / Milestone:** October 16, 2026  
**Status:** Alpha v0.1.0a1 (Ready for PyPI Packaging, CI/CD Active, 21/21 Unit Tests Passing)  
**Parent Suite:** Audio ML Data Curation Tier 2 (WaveWash ➔ SplitSafe ➔ SegDupe)

---

## 1. Executive Summary

**SplitSafe** is an open-source, local-first Python library and CLI auditing tool engineered to detect **data leakage** and **speaker contamination** across machine learning dataset splits (`train/`, `val/`, `test/`). 

In speech recognition (ASR/Whisper), text-to-speech (TTS/Voice Cloning), and speaker identification, models frequently appear to achieve state-of-the-art accuracy during training, only to severely degrade in production. The root cause is almost always **subtle dataset contamination**:
* The exact same recording (or a re-encoded, resampled, gain-adjusted, or trimmed copy) inadvertently placed in both `train` and `test` splits.
* The same speaker voice appearing across both training and evaluation splits, leading to over-optimistic evaluation metrics.

SplitSafe uses **deterministic spectral-peak landmark audio fingerprinting** (paired spectral peak hashing with time-offset alignment voting) to detect duplicate and partial audio across split boundaries in seconds without neural networks, GPU requirements, or external cloud services. Original files are never modified.

---

## 2. Key Links, Metadata & Ecosystem

| Resource | Official Reference / Destination | Status / Target |
| :--- | :--- | :--- |
| **PyPI Package** | [https://pypi.org/project/splitsafe/](https://pypi.org/project/splitsafe/) | **v0.1.0a1** (`pip install --pre splitsafe`) |
| **GitHub Repository** | [https://github.com/BoobalanArjunan/splitsafe](https://github.com/BoobalanArjunan/splitsafe) | Branch `main`, Tag `v0.1.0a1`, CI/CD Workflows |
| **License** | MIT License | Permissive open-source distribution |
| **Test Suite** | `tests/test_splits.py` | **21 / 21 Tests Passing (100%)** |
| **Reference Companion** | [WaveWash](https://pypi.org/project/wavewash/) | Tier 1 Data Preparation & QC Companion |

---

## 3. Core Architecture & Signal Processing

### A. Landmark Constellation Audio Fingerprinting
Unlike simple MD5 checksums that fail if an audio file is converted from FLAC to WAV or resampled by 1 Hz, SplitSafe operates on psychoacoustic and spectral invariants:
1. **Downsampling & Mono Conversion:** Audio is resampled via polyphase filtering (`signal.resample_poly`) to a standard $8\,\text{kHz}$ mono representation (`float32`).
2. **STFT Spectrogram Analysis:** 1024-sample window (128 ms) with hop size of 256 samples (32 ms) produces frequency bins up to 512.
3. **Local Peak Detection:** 2D maximum filter (`size=(21, 21)`) extracts dominant spectral peaks exceeding the 90th percentile noise floor, capped at 60 peaks per second to guard against dense music explosion.
4. **Target Zone Peak Pairing & Hashing:** Each peak is paired with up to 8 subsequent peaks within a 100-frame time window (~3.2 s). Packed 64-bit integer hashes are formed via:
   $$\text{Hash} = f_1 \mid (f_2 \ll 10) \mid (\Delta t \ll 20)$$
5. **Time-Offset Coincidence Voting:** Two recordings that share identical audio will produce clusters of identical hash keys with a **consistent time delta** ($\Delta T = t_{\text{file2}} - t_{\text{file1}}$). Random coincidences scatter evenly, whereas true audio copies create a massive spike in the time-offset histogram.
6. **Cross-Split Thresholding:** If aligned peak matches exceed `--min-votes` (default 30), a definitive leak is flagged with exact overlap percentage and file pair location.

### B. Speaker Contamination & Manifest Auditing
* **Regex Extraction:** Optional `--speaker-regex` extracts speaker identities directly from folder hierarchies or filenames.
* **Manifest Parsing:** Supports standard ML data CSV manifests with headers `file,split[,speaker]`.
* **Configurable Strictness:**
  * **Default (WARN):** Warns if identical speakers cross split boundaries (acceptable in some general ASR tasks).
  * **Strict Mode (`--strict-speakers` ➔ FAIL):** Fails CI/CD build if any speaker is leaked between train and test sets (mandatory for voice cloning and speaker verification).

---

## 4. Software Stack & CLI Tooling

### A. Technical Stack
* **Language:** Python 3.10+
* **Dependencies:** `numpy`, `scipy`, `soundfile` (zero heavy ML dependencies for lightweight, instantaneous execution).
* **CLI Entrypoint:** `splitsafe check`

### B. CLI Commands & Usage Patterns
```bash
# 1. Standard Folder Split Check
splitsafe check data/train data/val data/test

# 2. Strict Speaker Separation for Voice Cloning / Speaker ID
splitsafe check data/train data/val data/test --speaker-regex '^[^/]+/(spk[0-9]+)_' --strict-speakers

# 3. CSV Manifest Verification with Output Export
splitsafe check --manifest dataset.csv --out reports/ --json
```

### C. Automated Exit Codes for CI/CD Pipelines
* `0`: **PASS** (Clean dataset) or **WARN** (Shared speakers when `--strict-speakers` is disabled).
* `1`: **FAIL** (Cross-split audio leakage detected or strict speaker leak).
* `2`: **ERROR** (Usage error or corrupted file).

---

## 5. Global Talent Visa (GTV) Evidence & Strategic Positioning

SplitSafe forms **Tier 2** of the unified Audio Curation Suite for international talent recognition:

```
┌────────────────────────────────────────────────────────┐
│   Tier 1: WaveWash (File-Level QC & Repair)           │
│   • SNR, clipping repair, LUFS, spectrogram rolloff    │
├────────────────────────────────────────────────────────┤
│   Tier 2: SplitSafe (Split-Level Leakage Auditor)      │
│   • Cross-split copy & speaker contamination checker   │
├────────────────────────────────────────────────────────┤
│   Tier 3: SegDupe (Segment-Level Sub-Clip Matcher)    │
│   • Sub-segment fingerprinting & alignment engine      │
└────────────────────────────────────────────────────────┘
```

* **GTV Criterion 1 (Innovation & Technical Depth):** Demonstrates applied digital signal processing algorithms solving core machine learning reproducibility and evaluation contamination problems.
* **GTV Criterion 2 (Ecosystem Impact):** Complements WaveWash to create a professional-grade, multi-tool open-source portfolio published across PyPI, GitHub, and community forums.

---

## 6. Current Milestones & Checkpoint Status

- [x] Project architecture, algorithms, and packaging configured (`pyproject.toml`, `src/splitsafe/`).
- [x] Full unit test suite implemented (`tests/test_splits.py`) with 21/21 passing tests.
- [x] WaveWash reference checkpoint documented and archived.
- [x] Real-data benchmark validation on LibriSpeech (`dev-clean` vs `test-clean`): **0 false alarms, 100% planted leak detection (56/56)**. See [REAL_DATA_VALIDATION_REPORT.md](file:///Users/mymac/Desktop/Ai%202026%20Work/Antigravity%20Works/Splitsafe/project_plan/REAL_DATA_VALIDATION_REPORT.md).
- [ ] **Next Step (Tomorrow):** PyPI Upload (`twine upload --repository testpypi dist/*` -> `twine upload dist/*`) and Git release tag (`v0.1.0a1`).
- [ ] **Next Project:** SplitSafe Studio (Enterprise Web/Desktop UI with Wavesurfer.js leak playback, PDF compliance certificate, and Stripe billing).
- [ ] Community & developer documentation expansion.
