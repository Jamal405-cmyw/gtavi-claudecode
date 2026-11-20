"""Field-test harness end to end on a synthetic stand-in VOD (no network, mock VLM server)."""
import json, shutil
import pytest
from conftest import ROOT, needs_ffmpeg
import field_test as FT
import synth
from test_judge import Mock, A as answer


def test_spearman():
    assert FT.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert FT.spearman([1, 2, 3], [3, 2, 1]) == -1.0
    assert FT.spearman([1, 2], [1, 2]) is None and FT.spearman([1, 2, 3], [5, 5, 5]) is None


def test_baseline_editing_and_detector_are_frozen():
    import hashlib, subprocess
    for f in ("moments.py", "cut.py", "facecam.py", "common.py"):
        orig = subprocess.run(["git", "-C", str(ROOT), "show", f"67acdb4:scripts/{f}"], capture_output=True).stdout
        if not orig:
            pytest.skip("git history not available")
        mine = (ROOT / "bench" / "baseline" / f).read_bytes().replace(b"\r\n", b"\n")  # CRLF checkouts on Windows
        assert hashlib.sha256(mine).digest() == hashlib.sha256(orig).digest(), f


@needs_ffmpeg
def test_full_protocol_on_synthetic_vod(tmp_path, monkeypatch):
    monkeypatch.setattr(FT, "HERE", tmp_path / "bench")          # results under tmp, code from the repo
    (tmp_path / "bench").mkdir()
    shutil.copytree(ROOT / "bench" / "baseline", tmp_path / "bench" / "baseline")
    case = synth.build_case(tmp_path / "src", seed=41, dur=180, n_events=3, n_decoys=2)
    FT.main(["prepare", str(case / "source.mp4"), "--name", "t", "--minutes", "10",
             "--chat", str(case / "chat.json"), "--transcript", str(case / "transcript.json")])
    d = tmp_path / "bench" / "field" / "t"
    c = json.loads((d / "case.json").read_text())
    assert c["chat"] and c["code"]["bench/baseline/moments.py"]
    m = Mock([answer(8, 7, 7, 8, 0.9), answer(1, 1, 1, 1, 0.05)] * 10)
    try:
        FT.main(["detect", "t", "--clips", "3", "--cam", "0.7:0.05:0.28:0.3", "--judge-host", m.url])
    finally:
        m.close()
    picks = json.loads((d / "picks.json").read_text())
    assert all(len(picks[k]["picks"]) >= 1 and "error" not in picks[k] for k in "ABC")
    assert any(p.get("judge") for p in picks["C"]["picks"])
    FT.main(["render", "t", "--layout", "react", "--cam", "0.7:0.05:0.28:0.3", "--credit", "@test"])
    renders = json.loads((d / "renders.json").read_text())
    assert all(r.get("file") for r in renders.values()), renders
    assert all("source_start" in r for r in renders.values())
    assert renders["A_1"].get("reframe") is None and renders["B_1"].get("reframe")   # A = original editing code
    FT.main(["page", "t"])
    page = (d / "blind" / "rate.html").read_text()
    key = json.loads((d / "key.json").read_text())
    assert not (d / "blind" / "key.json").exists()
    for cfg in ("A_", "B_", "C_", "original", "improved", "VLM"):
        assert cfg not in page                                        # blind: no configuration names on the page
    ids = [it["id"] for it in key["items"]]
    assert all((d / "blind" / f"{i}.mp4").exists() for i in ids)
    # identical B/C spans are reviewed once
    assert len(ids) == len({r["file"] for r in renders.values()})
    ratings = {i: {"entertaining": 4, "setup": "yes", "payoff": "partly", "post": "yes", "start_ok": "yes", "end_ok": "no",
                   "facecam": "good", "gameplay": "good", "subs": "minor"} for i in ids}
    (d / "blind" / "ratings.json").write_text(json.dumps(ratings))
    res = FT.main(["score", "t", "--clips", "3"])
    rep = (d / "report.md").read_text()
    assert "## 1. Highlight selection" in rep and "## 2. Editing quality" in rep and "## 3. Processing time" in rep
    assert "unchanged between prepare and scoring" in rep
    log_rows = [ln for ln in rep.splitlines() if ln.startswith("| detect.") or ln.startswith("| render.")]
    assert log_rows and not any("%" in ln for ln in log_rows)        # regression: 0.4 s was printed as "40%"
    assert "| 100% |" in rep or "| 67% |" in rep or "| 33% |" in rep
    for cfg in "ABC":
        n = len(picks[cfg]["picks"][:3])
        assert res[cfg]["selection"]["precision_at_k"] == round(n / 3, 2)
        assert res[cfg]["editing"]["end_ok"] == 0.0 and res[cfg]["editing"]["subs"] == 0.5
    assert "vlm_effect" in res


@needs_ffmpeg
def test_missing_vlm_is_a_recorded_failure_not_a_silent_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(FT, "HERE", tmp_path / "bench")
    (tmp_path / "bench").mkdir()
    shutil.copytree(ROOT / "bench" / "baseline", tmp_path / "bench" / "baseline")
    case = synth.build_case(tmp_path / "src", seed=42, dur=120, n_events=2, n_decoys=1)
    FT.main(["prepare", str(case / "source.mp4"), "--name", "t", "--chat", str(case / "chat.json"),
             "--transcript", str(case / "transcript.json")])
    FT.main(["detect", "t", "--configs", "C", "--clips", "2", "--cam", "none", "--judge-host", "http://127.0.0.1:9"])
    d = tmp_path / "bench" / "field" / "t"
    picks = json.loads((d / "picks.json").read_text())
    assert "VLM judge did not run" in picks["C"]["error"]
    log = json.loads((d / "log.json").read_text())
    assert any(s["step"] == "detect.C.judge" and s["status"] == "failed" for s in log)


def test_code_change_is_flagged(tmp_path):
    hashes = FT.code_hashes()
    assert "scripts/highlight/engine.py" in hashes and "bench/baseline/cut.py" in hashes


def test_code_hashes_ignore_line_endings(tmp_path, monkeypatch):
    """Regression: on Windows (core.autocrlf=true) byte hashes differed from Linux for identical code."""
    (tmp_path / "x.py").write_bytes(b"a = 1\r\nb = 2\r\n")
    (tmp_path / "y.py").write_bytes(b"a = 1\nb = 2\n")
    monkeypatch.setattr(FT, "ROOT", tmp_path)
    monkeypatch.setattr(FT, "HASHED", ["x.py", "y.py"])
    h = FT.code_hashes()
    assert h["x.py"] == h["y.py"]

