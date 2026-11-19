"""Regression tests: one per fixed bug. Inputs are hand-built (numpy / ffmpeg lavfi), not the synthetic
episode generator the thresholds were tuned on, so these do not just re-test the tuning data."""
import inspect, os, subprocess, sys, wave
import numpy as np
import pytest
from conftest import ROOT, needs_ffmpeg
from highlight import accel, audio, candidates, engine, visual

SR = 16000


def _wav(path, x):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())
    return path


def _lavfi(path, *args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args, str(path)], check=True)
    return path


# Bug: steady noise / flat audio produced 5 "highlights" at 0.53 entertainment (z-scores of tiny wobbles).
def test_flat_noise_yields_no_candidates():
    x = np.random.default_rng(0).normal(0, 0.05, SR * 120).astype(np.float32)
    st = candidates.run(audio.features(x=x))
    assert st.shortlist == []


def test_flat_music_bed_yields_no_candidates_after_its_start():
    t = np.arange(SR * 120) / SR
    x = (0.2 * np.sin(2 * np.pi * 220 * t) * (0.7 + 0.3 * np.sin(2 * np.pi * 2 * t))).astype(np.float32)
    st = candidates.run(audio.features(x=x))
    assert all(p < 3 for p in st.shortlist)


def test_real_jump_still_detected_through_the_gate():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.02, SR * 90)
    x[SR * 50: SR * 53] += rng.normal(0, 0.3, SR * 3)  # +20 dB burst
    st = candidates.run(audio.features(x=x.astype(np.float32)))
    assert st.shortlist and 48 <= st.shortlist[0] <= 55


# Bug: a silent video still sampled a visual baseline (1.7 s) for zero candidates.
def test_no_visual_work_without_candidates(tmp_path, monkeypatch):
    wav = _wav(tmp_path / "a.wav", np.zeros(SR * 60))
    monkeypatch.setattr(visual, "baseline", lambda *a, **k: pytest.fail("baseline sampled with no candidates"))
    res = engine.analyse(str(tmp_path / "missing.mp4"), wav, [], 60.0, engine.Config(cam="none"), has_video=True)
    assert res["highlights"] == [] and res["timings"]["vis_frames"] == 0


# Bug: a video without an audio track crashed inside ffmpeg with an unhelpful error.
@needs_ffmpeg
def test_no_audio_track_exits_with_clear_message(tmp_path):
    v = _lavfi(tmp_path / "noaudio.mp4", "-f", "lavfi", "-i", "testsrc=s=160x90:r=5", "-t", "5", "-c:v", "libx264")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "highlights.py"), str(v), "--no-transcribe"],
                       env=dict(os.environ, CLIP_DESK_HOME=str(tmp_path)), capture_output=True, text=True, cwd=tmp_path)
    assert r.returncode != 0 and "no audio track" in r.stderr and "Traceback" not in r.stderr


def test_short_video_is_flagged(tmp_path):
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.01, SR * 10); x[SR * 5: SR * 6] += rng.normal(0, 0.4, SR)
    wav = _wav(tmp_path / "a.wav", x)
    res = engine.analyse(None, wav, [], 10.0, engine.Config(visual=False))
    assert any("shorter than --min" in n for n in res["notes"])
    assert all(h["end"] <= 10.0 for h in res["highlights"])


# GPU decode must fall back to CPU decode when the accelerator is missing or broken.
@needs_ffmpeg
def test_hwaccel_failure_falls_back_to_cpu(tmp_path, monkeypatch):
    v = _lavfi(tmp_path / "v.mp4", "-f", "lavfi", "-i", "testsrc=s=160x90:r=10", "-t", "4", "-c:v", "libx264")
    monkeypatch.setattr(accel, "decode_hwaccel", lambda pref="auto": "no_such_accel")
    fr = visual.frames(v, 0, 4, fps=2, hwaccel="auto")
    assert len(fr) >= 7


def test_whisper_choice_gpu_and_cpu(monkeypatch):
    monkeypatch.setattr(accel, "cuda_devices", lambda: 1)
    assert accel.whisper_choice() == ("large-v3-turbo", "cuda", "float16")
    monkeypatch.setattr(accel, "cuda_devices", lambda: 0)
    assert accel.whisper_choice() == ("small", "cpu", "int8")
    assert accel.whisper_choice("medium", "cuda") == ("medium", "cuda", "float16")


def test_moments_transcribe_defaults_unchanged():
    import moments
    p = inspect.signature(moments.transcribe).parameters
    assert p["model"].default == "small" and p["device"].default is None


def test_baseline_is_frozen():
    """The benchmark baseline must stay byte-identical to the original detector (commit 67acdb4)."""
    import hashlib
    h = hashlib.sha256((ROOT / "bench" / "baseline" / "moments.py").read_bytes()).hexdigest()
    assert h == BASELINE_SHA256


BASELINE_SHA256 = "04ce3cc8520810f468fccd10a2dfcb42b912f6031bab36ee76d9aaea2dc5f5c6"
