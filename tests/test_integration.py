"""End-to-end: synthetic labelled stream -> highlights.py CLI -> candidates.json -> cut.py renders a clip."""
import json, os, shutil, subprocess, sys
from pathlib import Path
import pytest
from conftest import ROOT, needs_ffmpeg
import synth


@pytest.fixture(scope="module")
def job(tmp_path_factory):
    home = tmp_path_factory.mktemp("home")
    case = synth.build_case(home / "case", seed=21, dur=150, n_events=2, n_decoys=1)
    env = dict(os.environ, CLIP_DESK_HOME=str(home))
    code = ("import sys; sys.path.insert(0, sys.argv[1]); from common import job_dir; print(job_dir(sys.argv[2]))")
    d = Path(subprocess.check_output([sys.executable, "-c", code, str(ROOT / "scripts"), str(case / "source.mp4")],
                                     env=env, text=True).strip())
    shutil.copy(case / "transcript.json", d / "transcript.json")
    shutil.copy(case / "chat.json", d / "chat.json")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "highlights.py"), str(case / "source.mp4"), "--top", "4",
                        "--cam", "0.7:0.05:0.28:0.3", "--hwaccel", "none", "--no-transcribe"],
                       env=env, capture_output=True, text=True, cwd=home)
    assert r.returncode == 0, r.stderr
    return d, json.loads((case / "labels.json").read_text()), env, home


@needs_ffmpeg
def test_highlights_find_labelled_events(job):
    d, labels, _, _ = job
    res = json.loads((d / "highlights.json").read_text())
    hl = res["highlights"]
    assert hl and res["disclaimer"]
    top2 = hl[:2]
    for lab in labels["highlights"]:
        assert any(h["start"] <= lab["peak"] <= h["end"] for h in top2), (lab, [(h["start"], h["end"]) for h in hl])
    for h in hl:
        assert 15 <= h["length"] <= 40 and h["evidence"] and 0 < h["confidence"] < 1
    assert res["signals"]["chat"] > 0 and res["signals"]["text"] > 0
    assert res["timings"]["vis_frames"] > 0


@needs_ffmpeg
def test_cut_py_consumes_new_candidates(job):
    d, _, env, home = job
    c = json.loads((d / "candidates.json").read_text())
    assert {"n", "start", "end", "score", "why", "text"} <= set(c["candidates"][0])
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "cut.py"), str(d), "--pick", "1", "--layout", "blur",
                        "--credit", "@test"], env=env, capture_output=True, text=True, cwd=home)
    assert r.returncode == 0, r.stderr[-800:]
    clip = d / "edits" / "clip_1.mp4"
    assert clip.exists() and clip.stat().st_size > 10_000
    chk = subprocess.run([sys.executable, str(ROOT / "scripts" / "check.py"), str(clip)], capture_output=True, text=True)
    assert "PASS  resolution 1080x1920" in chk.stdout and "PASS  has audio" in chk.stdout
