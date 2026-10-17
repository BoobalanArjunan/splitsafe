# splitsafe

[![tests](https://github.com/BoobalanArjunan/splitsafe/actions/workflows/tests.yml/badge.svg)](https://github.com/BoobalanArjunan/splitsafe/actions/workflows/tests.yml)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-yellow)](https://huggingface.co/spaces/boobalanit/splitsafe)
![status: alpha](https://img.shields.io/badge/status-alpha-orange)
![license: MIT](https://img.shields.io/badge/license-MIT-blue)

**Check your audio train / validation / test folders for leakage before you train.**

> **Core Architectural Finding:**  
> SplitSafe brings established spectral landmark fingerprinting techniques into an integrated, partition-aware audio ML dataset auditing workflow—guaranteeing zero evaluation contamination and speaker boundary integrity before model training.

If the same recording (or a re-encoded, resampled, louder, trimmed or clipped copy of it) sits on both sides of a split, your test score is measured on something the model has already seen, so it looks better than it really is. `splitsafe` finds those files, and also shows which speakers appear in more than one split.

It is small, runs on your own computer, never changes your files, and uses classic signal processing (spectral-peak fingerprints). There is no neural network and no model download.

## Install

```
pip install --pre splitsafe
```

(`--pre` is needed because this is an alpha release. Python 3.10 or newer.)

## Use

```
splitsafe check train/ val/ test/
```

Each folder is one split (subfolders are included). Example output from a small synthetic demo (two recordings were copied into the test folder, one resampled, one cut into a short clip):

```
splitsafe verdict: FAIL
Splits: test (4 files), train (5 files)

LEAKAGE: 2 file pair(s) appear on both sides of a split (4 files involved).
  [same audio (re-encoded / trimmed / re-leveled)] test/leaked_copy.wav  <->  train/spk9_a.wav   (aligned matches: 925, overlap: 100%)
  [same audio (re-encoded / trimmed / re-leveled)] test/leaked_clip.flac  <->  train/spk1_a.wav   (aligned matches: 93, overlap: 29%)

SPEAKERS in more than one split: 1
  spk0: train=1, test=1
```

The exit code is `0` for PASS or WARN, `1` for FAIL, `2` for a usage error, so it can run in CI.

### Speakers

Speaker overlap is only checked if you tell it who is who:

```
splitsafe check train val test --speaker-regex '^[^/]+/(spk[0-9]+)_'     # one capture group = the speaker id
splitsafe check --manifest data.csv                                     # columns: file,split[,speaker]
```

Shared speakers give a **WARN** (for speech recognition this is often acceptable). Add `--strict-speakers` to make it a **FAIL**, which is usually right for speaker ID or voice cloning.

### Options

| Option | What it does |
|---|---|
| `--names a,b,c` | Split names (default: the folder names) |
| `--manifest file.csv` | Use a CSV with `file,split[,speaker]` instead of folders. `--base DIR` sets what the file paths are relative to |
| `--out DIR` | Write `splitsafe_report.json`, `cross_split_duplicates.csv`, `splitsafe_report.txt`, and `splitsafe_report.html` |
| `--html PATH` | Write interactive HTML listening report with side-by-side audio players |
| `--json` | Print JSON instead of text |

## Use from Python

```python
from pathlib import Path
from splitsafe import check_splits
from splitsafe.scan import items_from_folders
from splitsafe.report import to_text

res = check_splits(items_from_folders([Path("train"), Path("test")]))
print(res.verdict())          # "PASS", "WARN" or "FAIL"
for m in res.matches:         # every leaked pair
    print(m.a, m.b, m.kind, m.votes)
print(to_text(res))
```

## What it checks

| Check | Finds |
|---|---|
| Exact copies | Byte-identical files in different splits |
| Same audio | The same recording after re-encoding (wav/flac), resampling, gain change, noise, trimming, or a short clip cut from a longer file |
| Shared speakers | Speakers (from your labels) that appear in more than one split |

Duplicates inside one split are not reported, because they are not leakage.

## Limitations (please read)

- **Alpha.** Tested only on synthetic audio so far (21 automatic tests, run on Linux, macOS and Windows in CI). **It has not yet been measured on real speech or music datasets.** The default threshold was chosen from synthetic data. On real data, tune `--min-votes` and check a few reported pairs by listening.
- **A PASS is not a guarantee.** Only the checks above are done. It cannot tell that two *different* recordings contain the same sentence, and it cannot find the same speaker without labels.
- **Heavy changes can hide a copy**: strong time-stretching, pitch-shifting, or very low-bitrate re-encoding may drop below the threshold.
- **Files that cannot be fingerprinted** (shorter than 1 second, silent, or unreadable) are listed under "NOT fully checked". Only exact byte copies are checked for them.
- **Speaker embeddings (finding the same voice without labels) are planned, not included.**
- The fingerprint idea (spectral peak pairs with time-offset voting) is a well-known technique. Chromaprint-based tools such as [audiomatch](https://pypi.org/project/audiomatch/) compare whole files. splitsafe's contribution is packaging it for train/test checks, including partial and clipped copies, with a clear verdict.

## How it works

Each file is turned into hashes of pairs of loud spectral peaks (frequency 1, frequency 2, time gap). Two files that share audio share many hashes **at one consistent time offset**. Random coincidences between different recordings do not line up in time, so counting aligned matches separates real copies from chance.

## Check it on real data

`scripts/real_data_check.py` runs splitsafe on two real, officially separate folders (for example LibriSpeech `dev-clean` and `test-clean`), reports any false alarms, plants changed copies of real files on the other side, and counts how many are found. See `docs/REAL_DATA_CHECK.md`.

## Development

```
pip install -e ".[dev]"
python -m pytest -q
```

## Contributing

Bug reports with a small example are the most useful. If splitsafe flags two files that are not the same audio, or misses a copy, please open an issue with the numbers from the report.

## License

MIT. See `LICENSE`.
