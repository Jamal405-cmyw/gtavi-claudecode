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
