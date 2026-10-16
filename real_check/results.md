# splitsafe real-data check

Folders: A = `/Users/mymac/LibriSpeech/LibriSpeech/dev-clean`, B = `/Users/mymac/LibriSpeech/LibriSpeech/test-clean`. Files: 2703 / 2620. min-votes = 30. seed = 0.

## 1. False alarms (A and B are officially separate)

- Cross-split pairs reported: **0**
- Shared speakers: **0**
- Files not fully checked: 0
- Highest vote counts among reported pairs: []
- Time: 55s

## 2. Planted leaks (real files, changed, put on the other side)

| Kind | planted | found | missed | min votes | median votes |
|---|---|---|---|---|---|
| exact copy | 10 | 10 | 0 | 306 | 764 |
| resampled 11k | 10 | 10 | 0 | 275 | 820 |
| quieter | 10 | 10 | 0 | 216 | 461 |
| trimmed | 10 | 10 | 0 | 80 | 369 |
| noisy ~20dB | 10 | 10 | 0 | 217 | 735 |
| 3 s clip | 6 | 6 | 0 | 76 | 214 |

**Total: 56 of 56 planted leaks found** (100%). Time: 55s.

If a kind has misses, lower `--min-votes` and run again, but check the false-alarm list in part 1 does not grow.
