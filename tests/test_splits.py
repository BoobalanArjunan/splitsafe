import csv
import json

import numpy as np
import pytest
import soundfile as sf

from conftest import SR, resample, tree_hashes, voice, write
from splitsafe.cli import main
from splitsafe.scan import apply_speaker_regex, check_splits, items_from_folders, items_from_manifest


def run(*folders, **kw):
    return check_splits(items_from_folders(list(folders)), **kw)


@pytest.fixture
def two(tmp_path):
    return tmp_path / "train", tmp_path / "test"


def test_different_recordings_pass(two):
    tr, te = two
    for i in range(10):
        write(tr / f"a{i}.wav", voice(100 + i))
        write(te / f"b{i}.wav", voice(900 + i))
    res = run(tr, te)
    assert res.verdict() == "PASS" and not res.matches and not res.skipped


def test_exact_copy_is_found(two):
    tr, te = two
    x = voice(1)
    write(tr / "a.wav", x)
    (te).mkdir()
    (te / "copy.wav").write_bytes((tr / "a.wav").read_bytes())
    write(te / "other.wav", voice(2))
    res = run(tr, te)
    assert res.verdict() == "FAIL"
    assert [m.kind for m in res.matches] == ["exact copy"]
    assert {res.matches[0].a, res.matches[0].b} == {"train/a.wav", "test/copy.wav"}


@pytest.mark.parametrize("variant", ["flac", "resampled", "quieter", "louder", "trimmed", "noisy"])
def test_changed_copy_is_found(two, variant):
    tr, te = two
    x = voice(7)
    write(tr / "orig.wav", x)
    for i in range(5):                                   # unrelated company so we also prove no false alarms
        write(tr / f"n{i}.wav", voice(300 + i))
        write(te / f"m{i}.wav", voice(400 + i))
    if variant == "flac":
        write(te / "v.flac", x)
    elif variant == "resampled":
        write(te / "v.wav", resample(x, 11025), 11025)
    elif variant == "quieter":
        write(te / "v.wav", x * 0.2)
    elif variant == "louder":
        write(te / "v.wav", np.clip(x * 3, -1, 1).astype(np.float32))
    elif variant == "trimmed":
        write(te / "v.wav", x[int(1.3 * SR):int(5.1 * SR)])
    elif variant == "noisy":
        write(te / "v.wav", (x + np.random.RandomState(1).randn(len(x)) * 0.02).astype(np.float32))
    res = run(tr, te)
    assert res.verdict() == "FAIL"
    assert len(res.matches) == 1, [(m.a, m.b, m.votes) for m in res.matches]
    assert {res.matches[0].a, res.matches[0].b} == {"train/orig.wav", "test/v." + ("flac" if variant == "flac" else "wav")}
    assert res.matches[0].kind.startswith("same audio")


def test_short_clip_cut_from_a_longer_file_is_found(two):
    tr, te = two
    long = voice(11, 12.0)
    write(tr / "long.wav", long)
    write(te / "clip.wav", long[int(4 * SR):int(7 * SR)])
    write(te / "other.wav", voice(12))
    res = run(tr, te)
    assert res.verdict() == "FAIL" and len(res.matches) == 1


def test_duplicates_inside_one_split_are_not_leakage(two):
    tr, te = two
    x = voice(5)
    write(tr / "a.wav", x)
    write(tr / "a_again.wav", x)
    write(te / "b.wav", voice(6))
    res = run(tr, te)
    assert res.verdict() == "PASS"


def test_three_splits_are_all_compared(tmp_path):
    x = voice(21)
    write(tmp_path / "train" / "a.wav", voice(22))
    write(tmp_path / "val" / "v.wav", x)
    write(tmp_path / "test" / "t.wav", x * 0.5)
    res = check_splits(items_from_folders([tmp_path / "train", tmp_path / "val", tmp_path / "test"]))
    assert res.verdict() == "FAIL"
    assert {(m.split_a, m.split_b) for m in res.matches} == {("test", "val")}


def test_short_or_silent_files_are_reported_not_silently_passed(two):
    tr, te = two
    write(tr / "ok.wav", voice(31))
    write(te / "ok2.wav", voice(32))
    write(te / "tiny.wav", np.zeros(100, np.float32) + 0.1)
    write(te / "silent.wav", np.zeros(SR * 3, np.float32))
    res = run(tr, te)
    assert {i.path for i in res.skipped} == {"test/tiny.wav", "test/silent.wav"}
    from splitsafe.report import to_text
    assert "NOT fully checked: 2" in to_text(res)


def test_exact_copy_of_a_short_file_is_still_found(two):
    tr, te = two
    write(tr / "tiny.wav", np.zeros(100, np.float32) + 0.1)
    te.mkdir()
    (te / "tiny2.wav").write_bytes((tr / "tiny.wav").read_bytes())
    write(te / "ok.wav", voice(33))
    res = run(tr, te)
    assert res.verdict() == "FAIL" and res.matches[0].kind == "exact copy"


def test_unreadable_file_is_reported(two):
    tr, te = two
    write(tr / "ok.wav", voice(41))
    te.mkdir()
    (te / "broken.wav").write_bytes(b"this is not audio")
    write(te / "ok2.wav", voice(42))
    res = run(tr, te)
    assert [i.path for i in res.skipped] == ["test/broken.wav"]
    assert "could not read" in res.skipped[0].skipped


def test_inputs_are_never_modified(two):
    tr, te = two
    write(tr / "a.wav", voice(51))
    write(te / "b.wav", voice(51))
    before = (tree_hashes(tr), tree_hashes(te))
    run(tr, te)
    assert before == (tree_hashes(tr), tree_hashes(te))


def test_speaker_overlap_from_manifest(tmp_path):
    for sp, split, seed in [("spk1", "train", 61), ("spk2", "train", 62), ("spk1", "test", 63), ("spk3", "test", 64)]:
        write(tmp_path / split / f"{sp}_{seed}.wav", voice(seed))
    m = tmp_path / "m.csv"
    with open(m, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "split", "speaker"])
        for sp, split, seed in [("spk1", "train", 61), ("spk2", "train", 62), ("spk1", "test", 63), ("spk3", "test", 64)]:
            w.writerow([f"{split}/{sp}_{seed}.wav", split, sp])
    res = check_splits(items_from_manifest(m))
    assert res.shared_speakers == {"spk1": {"train": 1, "test": 1}}
    assert res.verdict() == "WARN" and res.verdict(strict_speakers=True) == "FAIL"
    assert not res.matches


def test_speaker_regex(two):
    tr, te = two
    write(tr / "alice_1.wav", voice(71))
    write(te / "alice_2.wav", voice(72))
    write(te / "bob_1.wav", voice(73))
    items = items_from_folders([tr, te])
    apply_speaker_regex(items, r"^[^/]+/([^_]+)_")
    res = check_splits(items)
    assert list(res.shared_speakers) == ["alice"]


def test_speaker_check_is_off_without_labels(two):
    tr, te = two
    write(tr / "a.wav", voice(81))
    write(te / "b.wav", voice(82))
    res = run(tr, te)
    assert not res.speakers_checked
    from splitsafe.report import to_text
    assert "NOT checked" in to_text(res)


def test_cli_exit_codes_and_reports(two, tmp_path, capsys):
    tr, te = two
    x = voice(91)
    write(tr / "a.wav", x)
    write(te / "b.wav", voice(92))
    assert main(["check", str(tr), str(te)]) == 0
    write(te / "dup.wav", x)
    out = tmp_path / "rep"
    assert main(["check", str(tr), str(te), "--out", str(out)]) == 1
    data = json.loads((out / "splitsafe_report.json").read_text())
    assert data["verdict"] == "FAIL" and len(data["cross_split_duplicates"]) == 1
    rows = list(csv.reader(open(out / "cross_split_duplicates.csv")))
    assert len(rows) == 2 and rows[0][0] == "file_a"
    assert (out / "splitsafe_report.html").exists()
    html_out = (out / "splitsafe_report.html").read_text()
    assert "<audio" in html_out
    assert "Play Both Aligned" in html_out

    custom_html = tmp_path / "custom_report.html"
    assert main(["check", str(tr), str(te), "--html", str(custom_html)]) == 1
    assert custom_html.exists()
    assert "<audio" in custom_html.read_text()

    capsys.readouterr()
    assert main(["check", str(tr), str(te), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["verdict"] == "FAIL"



def test_cli_usage_errors(two, capsys):
    tr, te = two
    write(tr / "a.wav", voice(95))
    assert main(["check", str(tr)]) == 2
    assert main(["check", str(tr), str(tr / "nope")]) == 2
    assert main([]) == 2
    assert main(["check", str(tr), str(te), "--names", "only_one"]) == 2


def test_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert "splitsafe" in capsys.readouterr().out
