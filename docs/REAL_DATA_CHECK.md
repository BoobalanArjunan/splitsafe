# Real-data check (LibriSpeech)

This tests splitsafe on real speech. LibriSpeech `dev-clean` and `test-clean` are officially separate, so they make a good test.

1. Download `dev-clean.tar.gz` and `test-clean.tar.gz` from https://www.openslr.org/12 (about 340 MB each) and unpack both into one folder, e.g. `~/LibriSpeech/`.
2. Install: `pip install --pre splitsafe` (use a project checkout if you want the `scripts/` folder).
3. Run: `python scripts/real_data_check.py --a ~/LibriSpeech/dev-clean --b ~/LibriSpeech/test-clean --plant 60 --out real_check`
4. Open `real_check/results.md`.

What the numbers mean:

- **Part 1, false alarms:** the two folders are separate, so ideally 0 pairs and 0 shared speakers. If pairs are listed, listen to a few. Real false alarms mean the threshold needs raising.
- **Part 2, planted leaks:** each kind of change should be found. If a kind has misses, try a lower `--min-votes` and re-run, while checking Part 1 stays clean.

Your original files are never modified. Planted copies are written under `real_check/planted/`.
Add `--max-files 300` for a quick trial before the full run.
