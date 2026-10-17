---
title: SplitSafe - Audio ML Leakage Auditor
emoji: 🛡️
colorFrom: red
colorTo: green
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
license: mit
short_description: Detect audio train/val/test data leakage & speaker overlap
---

# SplitSafe: Audio ML Dataset Leakage & Contamination Auditor

Check your audio **train / validation / test** splits for data leakage before wasting expensive GPU compute on false benchmarks.

### Features
- **100% Signal Processing:** Fast, zero-GPU deterministic landmark constellation audio fingerprinting.
- **Audio-Invariant:** Catches copies even if re-encoded, resampled, quieter, trimmed, or short sub-clips.
- **Speaker Contamination Guard:** Detects if the same speaker appears on both sides of a split.

*Created by **Boobalan Arjunan** ([boobalan.it@gmail.com](mailto:boobalan.it@gmail.com))*
