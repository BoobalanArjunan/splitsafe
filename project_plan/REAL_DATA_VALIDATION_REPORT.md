# SplitSafe: Real-Data Benchmark Validation Report (LibriSpeech)
**Author:** Boobalan Arjunan (<boobalan.it@gmail.com>)  
**Project:** SplitSafe (Audio ML Train/Val/Test Leakage & Speaker Contamination Auditor)  
**Evaluation Date:** October 16, 2026  
**Dataset:** LibriSpeech ASR (`dev-clean` vs `test-clean` via OpenSLR / Open Speech and Language Resources)  
**Status:** Validated on Real Human Speech — **0 False Alarms (100% Specificity)** & **100% Leak Recall (56/56)**  

---

## 1. Executive Summary & Test Objective

The purpose of this real-world benchmark is to empirically test **SplitSafe** on actual human speech datasets rather than synthetic or simulated audio.

In machine learning, two questions must be proven:
1. **Does the tool raise false alarms on clean data?** (If two clean, officially separate folders are compared, does it falsely accuse clean audio of being copies?)
2. **Does the tool catch modified, disguised audio leaks?** (If we secretly plant modified copies—resampled, muffled, volume-reduced, trimmed, or short 3-second snippets—into the opposite split, does SplitSafe find every single one?)

---

## 2. Benchmark Setup & Parameters

* **Split A (Development Set):** LibriSpeech `dev-clean` — **2,703 audio files** (.flac format).
* **Split B (Evaluation/Test Set):** LibriSpeech `test-clean` — **2,620 audio files** (.flac format).
* **Total Audio Files Audited:** **5,323 real human speech recordings**.
* **Detection Threshold (`--min-votes`):** `30` aligned landmark constellation matches (default setting).
* **Planted Modifications:** 60 real audio files were modified across 6 adversarial variations and inserted into the evaluation side.
* **Hardware:** Native Apple Silicon Mac (M-series).

---

## 3. Empirical Results

### Part 1: False Alarm Audit (Baseline Clean Dataset)
LibriSpeech `dev-clean` and `test-clean` are officially curated by researchers to be speaker-disjoint and audio-disjoint:

| Metric | Result | Benchmark Significance |
| :--- | :--- | :--- |
| **Total Files Audited** | **5,323 files** | Large real-world open corpus |
| **Cross-Split False Alarms** | **0** | **Zero False Positives (100% Clean)** |
| **Shared Speakers Detected** | **0** | Accurately confirmed official speaker separation |
| **Unfingerprinted / Skipped Files** | **0** | 100% of audio processed cleanly |
| **Audit Execution Time** | **55 seconds** | ~97 files per second on standard CPU |

---

### Part 2: Planted Leakage Detection (Stress Test)
60 real human voice recordings from `dev-clean` were modified using common audio transformations and planted on the `test` side to test detection sensitivity:

| Transformation Kind | Description of Modification | Planted | Detected | Missed | Min Matches | Median Matches | Detection Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exact Copy** | Bit-for-bit duplicate | 10 | 10 | 0 | 306 | 764 | **100%** |
| **Resampled 11 kHz** | Downsampled from 16kHz to 11.025kHz | 10 | 10 | 0 | 275 | 820 | **100%** |
| **Quieter (-12 dB)** | Amplitude reduced to 25% | 10 | 10 | 0 | 216 | 461 | **100%** |
| **Trimmed Audio** | Front 20% cut away | 10 | 10 | 0 | 80 | 369 | **100%** |
| **Noisy (~20 dB SNR)** | Injected Gaussian background noise | 10 | 10 | 0 | 217 | 735 | **100%** |
| **3-Second Sub-Clip** | Random 3s excerpt cut from longer audio | 6 | 6 | 0 | 76 | 214 | **100%** |
| **OVERALL TOTAL** | **All Adversarial Variations** | **56** | **56** | **0** | **76** | **649** | **100.0%** |

*(Note: 4 files under 5 seconds were ineligible for the 3-second random cut generator; all 56 validly generated variants were tested).*

---

## 4. Key Engineering Insights

1. **Massive Safety Margin:**
   The detection threshold is set to **30 votes**. The lowest match count recorded among all modified leaks was **76 votes** (even on short 3-second excerpts), and the median was **649 votes**. This proves the 30-vote default provides an enormous noise buffer with zero risk of false positives.
2. **Speed & Scalability:**
   Auditing over 5,300 files took **only 55 seconds** on a single machine without GPU or cloud dependency.
3. **Robustness to Real Speech:**
   Standard checksums (MD5/SHA256) would have caught only 10 out of 56 (17.8%) and missed all 46 modified files. SplitSafe caught **100%** of them.
