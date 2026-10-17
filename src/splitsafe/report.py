import csv
import html
import json
import os
from pathlib import Path

from .scan import Result

NOTE = ("Note: a PASS means none of these checks found a problem. It is not a guarantee: only exact copies, "
        "same-audio copies and the speaker labels you provided are checked.")


def to_dict(res: Result, strict_speakers=False):
    def _clean_match(m):
        d = dict(m.__dict__)
        if "abs_a" in d:
            d["abs_a"] = str(d["abs_a"]) if d["abs_a"] is not None else None
        if "abs_b" in d:
            d["abs_b"] = str(d["abs_b"]) if d["abs_b"] is not None else None
        return d

    return {
        "verdict": res.verdict(strict_speakers),
        "splits": res.splits,
        "min_votes": res.min_votes,
        "cross_split_duplicates": [_clean_match(m) for m in res.matches],
        "shared_speakers": res.shared_speakers if res.speakers_checked else None,
        "speakers_checked": res.speakers_checked,
        "files_not_checked": [{"file": i.path, "reason": i.skipped} for i in res.skipped],
        "note": NOTE,
    }



def to_text(res: Result, strict_speakers=False, limit=15) -> str:
    v = res.verdict(strict_speakers)
    L = [f"splitsafe verdict: {v}", "Splits: " + ", ".join(f"{k} ({n} files)" for k, n in res.splits.items()), ""]
    if res.matches:
        L.append(f"LEAKAGE: {len(res.matches)} file pair(s) appear on both sides of a split "
                 f"({len(res.leaked_files)} files involved).")
        for m in res.matches[:limit]:
            L.append(f"  [{m.kind}] {m.a}  <->  {m.b}   (aligned matches: {m.votes}, overlap: {m.coverage:.0%})")
        if len(res.matches) > limit:
            L.append(f"  ... and {len(res.matches) - limit} more (see the report file)")
        L.append("")
    else:
        L.append("No duplicate audio found across splits.")
        L.append("")
    if res.speakers_checked:
        if res.shared_speakers:
            L.append(f"SPEAKERS in more than one split: {len(res.shared_speakers)}")
            for s, d in list(res.shared_speakers.items())[:limit]:
                L.append(f"  {s}: " + ", ".join(f"{k}={n}" for k, n in d.items()))
            if len(res.shared_speakers) > limit:
                L.append(f"  ... and {len(res.shared_speakers) - limit} more")
            L.append("  (For speech recognition this can be acceptable; for speaker ID or voice cloning it is usually a leak.)")
            L.append("")
        else:
            L.append("No speaker appears in more than one split.")
            L.append("")
    else:
        L.append("Speaker overlap was NOT checked (no speaker labels). Use --manifest with a speaker column or --speaker-regex.")
        L.append("")
    if res.skipped:
        L.append(f"NOT fully checked: {len(res.skipped)} file(s)")
        for i in res.skipped[:5]:
            L.append(f"  {i.path}: {i.skipped}")
        if len(res.skipped) > 5:
            L.append(f"  ... and {len(res.skipped) - 5} more")
        L.append("")
    L.append(NOTE)
    return "\n".join(L)


def to_html(res: Result, out_path: Path | None = None, strict_speakers: bool = False) -> str:
    verdict = res.verdict(strict_speakers)
    if verdict == "PASS":
        badge_style = "background: linear-gradient(135deg, #10b981, #059669); color: white;"
        badge_text = "🟢 PASS &mdash; NO LEAKAGE DETECTED"
        summary_desc = "All dataset splits are clean and isolated. Safe to proceed with model training."
    elif verdict == "WARN":
        badge_style = "background: linear-gradient(135deg, #f59e0b, #d97706); color: white;"
        badge_text = "🟡 WARN &mdash; SHARED SPEAKERS DETECTED"
        summary_desc = "Shared speakers found across partitions. Verify if your downstream task permits speaker overlap."
    else:
        badge_style = "background: linear-gradient(135deg, #ef4444, #dc2626); color: white;"
        badge_text = "🔴 FAIL &mdash; DATA LEAKAGE DETECTED"
        summary_desc = "Critical evaluation contamination! Audio from your training set was found in your evaluation set."

    def _resolve_src(abs_path: Path | None, fallback: str) -> str:
        if abs_path:
            p = Path(abs_path)
            if p.is_file():
                if out_path:
                    try:
                        return os.path.relpath(p, Path(out_path).parent)
                    except Exception:
                        pass
                try:
                    return p.as_uri()
                except Exception:
                    return str(p)
        return fallback

    matches_html = []
    for i, m in enumerate(res.matches):
        src_a = _resolve_src(m.abs_a, m.a)
        src_b = _resolve_src(m.abs_b, m.b)
        overlap_pct = f"{m.coverage:.0%}"
        matches_html.append(f"""
        <div class="match-item">
            <div class="match-header">
                <div class="match-title">
                    <span class="match-index">#{i + 1}</span>
                    <span class="file-name">{html.escape(m.a)}</span>
                    <span class="arrow">&harr;</span>
                    <span class="file-name">{html.escape(m.b)}</span>
                </div>
                <div class="match-badges">
                    <span class="badge badge-kind">{html.escape(m.kind)}</span>
                    <span class="badge badge-votes">{m.votes} votes ({overlap_pct} overlap)</span>
                </div>
            </div>
            <div class="player-grid">
                <div class="player-card">
                    <div class="player-meta">
                        <span class="split-badge">{html.escape(m.split_a)}</span>
                        <span class="player-path" title="{html.escape(m.a)}">{html.escape(m.a)}</span>
                    </div>
                    <audio id="audio_{i}_a" controls preload="none">
                        <source src="{html.escape(src_a)}">
                        Your browser does not support audio.
                    </audio>
                </div>
                <div class="player-card">
                    <div class="player-meta">
                        <span class="split-badge split-b">{html.escape(m.split_b)}</span>
                        <span class="player-path" title="{html.escape(m.b)}">{html.escape(m.b)}</span>
                    </div>
                    <audio id="audio_{i}_b" controls preload="none">
                        <source src="{html.escape(src_b)}">
                        Your browser does not support audio.
                    </audio>
                </div>
            </div>
            <div class="sync-actions">
                <button type="button" class="btn-sync" onclick="playPair({i})">⚡ Play Both Aligned</button>
                <button type="button" class="btn-stop" onclick="stopPair({i})">⏹️ Stop</button>
            </div>
        </div>
        """)

    matches_section = "".join(matches_html) if matches_html else '<div class="empty-state">🎉 No duplicate audio or leaked pairs found across partitions!</div>'

    speakers_html = ""
    if res.speakers_checked and res.shared_speakers:
        rows = "".join(f"<tr><td><code>{html.escape(s)}</code></td><td>" + ", ".join(f"{html.escape(k)}={n}" for k, n in d.items()) + "</td></tr>" for s, d in res.shared_speakers.items())
        speakers_html = f"""
        <div class="section-card">
            <h3>👥 Shared Speaker Identities Across Splits ({len(res.shared_speakers)})</h3>
            <table class="data-table">
                <thead><tr><th>Speaker ID</th><th>Files per Split</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        """

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SplitSafe Listening Audit Report</title>
    <style>
        :root {{
            --bg: #090d16;
            --surface: #111827;
            --surface-hover: #1f2937;
            --border: rgba(255, 255, 255, 0.08);
            --text: #f8fafc;
            --muted: #94a3b8;
            --primary: #f97316;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg);
            color: var(--text);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            padding: 30px 20px;
            line-height: 1.5;
        }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        .header {{
            background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.4);
        }}
        .header-top {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 12px; }}
        .brand {{ font-size: 22px; font-weight: 800; display: flex; align-items: center; gap: 8px; }}
        .brand span {{ color: var(--primary); }}
        .verdict-badge {{ padding: 6px 14px; border-radius: 9999px; font-size: 13px; font-weight: 700; {badge_style} }}
        .summary-desc {{ color: var(--muted); font-size: 14px; margin-bottom: 16px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }}
        .stat-card {{ background: rgba(15, 23, 42, 0.6); border: 1px solid var(--border); padding: 12px 16px; border-radius: 10px; }}
        .stat-label {{ font-size: 11px; text-transform: uppercase; color: var(--muted); font-weight: 600; }}
        .stat-val {{ font-size: 20px; font-weight: 700; margin-top: 4px; color: #f1f5f9; }}
        .match-item {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-left: 4px solid var(--primary);
            border-radius: 12px;
            padding: 18px;
            margin-bottom: 16px;
        }}
        .match-header {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }}
        .match-title {{ font-size: 14px; font-family: monospace; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
        .match-index {{ background: rgba(249, 115, 22, 0.2); color: var(--primary); padding: 2px 8px; border-radius: 6px; font-weight: 700; }}
        .file-name {{ color: #e2e8f0; font-weight: 600; }}
        .arrow {{ color: var(--muted); }}
        .badge {{ padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 600; }}
        .badge-kind {{ background: rgba(56, 189, 248, 0.15); color: #7dd3fc; border: 1px solid rgba(56, 189, 248, 0.3); }}
        .badge-votes {{ background: rgba(239, 68, 68, 0.15); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.3); }}
        .player-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-bottom: 12px; }}
        .player-card {{ background: rgba(0, 0, 0, 0.2); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }}
        .player-meta {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }}
        .split-badge {{ background: #2563eb; color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px; font-weight: 700; text-transform: uppercase; }}
        .split-b {{ background: #dc2626; }}
        .player-path {{ font-size: 12px; color: var(--muted); font-family: monospace; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        audio {{ width: 100%; height: 36px; border-radius: 6px; }}
        .sync-actions {{ display: flex; gap: 8px; }}
        .btn-sync, .btn-stop {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            color: #cbd5e1;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.15s;
        }}
        .btn-sync:hover {{ background: rgba(249, 115, 22, 0.2); color: white; border-color: var(--primary); }}
        .btn-stop:hover {{ background: rgba(239, 68, 68, 0.2); color: white; }}
        .section-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 20px; margin-top: 24px; }}
        .section-card h3 {{ font-size: 16px; margin-bottom: 12px; color: #f8fafc; }}
        .data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }}
        .data-table th, .data-table td {{ padding: 10px; border-bottom: 1px solid var(--border); }}
        .data-table th {{ color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 11px; }}
        .empty-state {{ text-align: center; padding: 40px; color: #10b981; font-weight: 600; font-size: 16px; background: var(--surface); border: 1px solid var(--border); border-radius: 12px; }}
        .footer {{ text-align: center; color: #64748b; font-size: 12px; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-top">
                <div class="brand">🛡️ SplitSafe <span>Listening Report</span></div>
                <div class="verdict-badge">{badge_text}</div>
            </div>
            <div class="summary-desc">{summary_desc}</div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">Partitions</div>
                    <div class="stat-val">{len(res.splits)} splits</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Leaked Pairs</div>
                    <div class="stat-val" style="color: {'#ef4444' if len(res.matches) > 0 else '#10b981'};">{len(res.matches)} pairs</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Contaminated Files</div>
                    <div class="stat-val" style="color: {'#ef4444' if len(res.leaked_files) > 0 else '#10b981'};">{len(res.leaked_files)} files</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Shared Speakers</div>
                    <div class="stat-val">{len(res.shared_speakers) if res.speakers_checked else 'N/A'}</div>
                </div>
            </div>
        </div>

        <h2 style="font-size: 18px; margin-bottom: 14px; color: #f8fafc;">🎧 Flagged Audio Pairs with Players ({len(res.matches)})</h2>
        {matches_section}

        {speakers_html}

        <div class="footer">
            Generated by <strong>SplitSafe v0.1.0a1</strong> &bull; Deterministic Signal Processing Audio ML Auditor
        </div>
    </div>

    <script>
        function playPair(idx) {{
            const a = document.getElementById('audio_' + idx + '_a');
            const b = document.getElementById('audio_' + idx + '_b');
            if (a && b) {{
                a.currentTime = 0;
                b.currentTime = 0;
                a.play();
                b.play();
            }}
        }}
        function stopPair(idx) {{
            const a = document.getElementById('audio_' + idx + '_a');
            const b = document.getElementById('audio_' + idx + '_b');
            if (a) {{ a.pause(); a.currentTime = 0; }}
            if (b) {{ b.pause(); b.currentTime = 0; }}
        }}
    </script>
</body>
</html>
"""
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(doc, encoding="utf-8")
    return doc


def write_reports(res: Result, out: Path, strict_speakers=False):
    out.mkdir(parents=True, exist_ok=True)
    (out / "splitsafe_report.json").write_text(json.dumps(to_dict(res, strict_speakers), indent=2))
    with open(out / "cross_split_duplicates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_a", "split_a", "file_b", "split_b", "kind", "aligned_matches", "overlap"])
        for m in res.matches:
            w.writerow([m.a, m.split_a, m.b, m.split_b, m.kind, m.votes, m.coverage])
    (out / "splitsafe_report.txt").write_text(to_text(res, strict_speakers, limit=10**9))
    to_html(res, out / "splitsafe_report.html", strict_speakers)
