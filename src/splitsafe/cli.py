from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .report import to_dict, to_text, write_reports
from .scan import (DEFAULT_MIN_VOTES, apply_speaker_regex, check_splits,
                   items_from_folders, items_from_manifest)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="splitsafe", description="Check audio train/val/test splits for leakage.")
    ap.add_argument("--version", action="version", version=f"splitsafe {__version__}")
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("check", help="compare two or more split folders (or a manifest)")
    c.add_argument("folders", nargs="*", help="one folder per split, e.g. train val test")
    c.add_argument("--names", help="split names, comma separated (default: folder names)")
    c.add_argument("--manifest", help="CSV with columns file,split[,speaker] instead of folders")
    c.add_argument("--base", help="folder the manifest's file paths are relative to (default: the CSV's folder)")
    c.add_argument("--speaker-regex", help="regex with ONE capture group that extracts the speaker id from the path "
                                           "shown in the report, e.g. '^[^/]+/([^/_-]+)'")
    c.add_argument("--strict-speakers", action="store_true", help="treat shared speakers as FAIL, not WARN")
    c.add_argument("--min-votes", type=int, default=DEFAULT_MIN_VOTES,
                   help=f"aligned fingerprint matches needed to call two files the same audio (default {DEFAULT_MIN_VOTES})")
    c.add_argument("--out", help="folder to write report files (json, csv, txt)")
    c.add_argument("--json", action="store_true", help="print JSON instead of text")
    a = ap.parse_args(argv)
    if a.cmd != "check":
        ap.print_help()
        return 2
    try:
        if a.manifest:
            items = items_from_manifest(Path(a.manifest), Path(a.base) if a.base else None)
        else:
            if len(a.folders) < 2:
                print("error: give at least two split folders, or use --manifest", file=sys.stderr)
                return 2
            names = a.names.split(",") if a.names else None
            if names and len(names) != len(a.folders):
                print("error: --names must have one name per folder", file=sys.stderr)
                return 2
            items = items_from_folders([Path(f) for f in a.folders], names)
        if a.speaker_regex:
            apply_speaker_regex(items, a.speaker_regex)
        if not items:
            print("error: no audio files found", file=sys.stderr)
            return 2

        def prog(n, total):
            if sys.stderr.isatty():
                print(f"\rreading {n}/{total}", end="", file=sys.stderr, flush=True)
        res = check_splits(items, a.min_votes, prog)
        if sys.stderr.isatty():
            print(file=sys.stderr)
    except (ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.out:
        write_reports(res, Path(a.out), a.strict_speakers)
    if a.json:
        import json
        print(json.dumps(to_dict(res, a.strict_speakers), indent=2))
    else:
        print(to_text(res, a.strict_speakers))
    return 1 if res.verdict(a.strict_speakers) == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
