"""Windows CUDA DLL registration and the GPU check helpers (no GPU needed)."""
import os, types
import cuda_dlls, gpu_check


def test_register_is_noop_off_windows(monkeypatch):
    monkeypatch.setattr(cuda_dlls, "_added", None)
    monkeypatch.setattr(cuda_dlls, "IS_WINDOWS", False)
    assert cuda_dlls.register() == []


def test_register_adds_nvidia_bin_folders_on_windows(tmp_path, monkeypatch):
    import sys
    for lib in ("cublas", "cudnn"):
        (tmp_path / "nvidia" / lib / "bin").mkdir(parents=True)
        (tmp_path / "nvidia" / lib / "bin" / f"{lib}64_x.dll").write_bytes(b"")
    (tmp_path / "nvidia" / "empty" / "bin").mkdir(parents=True)          # no DLLs: ignored
    fake = types.ModuleType("nvidia"); fake.__path__ = [str(tmp_path / "nvidia")]
    monkeypatch.setitem(sys.modules, "nvidia", fake)
    added = []
    monkeypatch.setattr(cuda_dlls, "IS_WINDOWS", True)
    monkeypatch.setattr(cuda_dlls.os, "add_dll_directory", lambda p: added.append(p), raising=False)
    monkeypatch.setattr(cuda_dlls, "_added", None)
    monkeypatch.setenv("PATH", "orig")
    got = cuda_dlls.register()
    assert [os.path.basename(os.path.dirname(p)) for p in got] == ["cublas", "cudnn"] and added == got
    assert os.environ["PATH"].endswith("orig") and got[0] in os.environ["PATH"]
    assert cuda_dlls.register() is got                                    # cached, PATH not grown twice
    monkeypatch.setattr(cuda_dlls, "_added", None)


def test_wer():
    assert gpu_check.wer("No way, he jumped the bridge!", "no way he jumped the bridge") == 0.0
    assert gpu_check.wer("one two three four", "one two four") == 0.25
    assert gpu_check.wer("a b", "a b c d") == 1.0
    assert gpu_check.wer("", "x") is None


def test_pipeline_never_passes_a_file_path_to_faster_whisper(tmp_path, monkeypatch):
    """Regression (found on the Windows PC): faster-whisper 1.2.1 decodes files with
    av.open(metadata_errors=...), which PyAV 19 rejects. moments.transcribe must pass samples."""
    import numpy as np, sys, types, wave
    import moments
    wav = tmp_path / "a.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(np.zeros(16000, np.int16).tobytes())
    seen = []

    class FakeModel:
        def __init__(self, *a, **k):
            pass

        def transcribe(self, audio, **k):
            seen.append(type(audio))
            return iter([]), None
    fw = types.ModuleType("faster_whisper"); fw.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fw)
    moments.transcribe(wav, tmp_path, model="tiny", device="cpu")
    assert seen == [np.ndarray]


def test_gpu_check_load_pcm(tmp_path):
    import subprocess
    p = tmp_path / "s.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=300:d=2", "-ar", "44100", "-ac", "2", str(p)], check=True)
    x = gpu_check.load_pcm(p)
    assert x.dtype.name == "float32" and abs(len(x) - 32000) <= 160 and 0.1 < abs(x).max() <= 1.0
