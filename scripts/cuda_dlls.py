"""Windows only: let CTranslate2 (faster-whisper) find cuBLAS/cuDNN installed as pip packages.

The ctranslate2 Windows wheel ships only a small cudnn64_9.dll loader and no cuBLAS. The NVIDIA pip
packages (`requirements-gpu-windows.txt`: nvidia-cublas-cu12, nvidia-cudnn-cu12) put the real DLLs in
<venv>/Lib/site-packages/nvidia/<lib>/bin. This adds those folders to the DLL search path of the
current process only (os.add_dll_directory + PATH for libraries that load their own dependencies).
Nothing system-wide changes. No-op on Linux/macOS or when the packages are not installed.
"""
import os
from pathlib import Path

_added = None
IS_WINDOWS = os.name == "nt"


def register():
    """Return the list of folders added (cached after the first call)."""
    global _added
    if _added is not None:
        return _added
    _added = []
    if not IS_WINDOWS:
        return _added
    try:
        import nvidia  # namespace package created by the nvidia-* wheels
    except ImportError:
        return _added
    for base in list(getattr(nvidia, "__path__", [])):
        for d in sorted(Path(base).glob("*/bin")) + sorted(Path(base).glob("*/lib")):
            if d.is_dir() and any(d.glob("*.dll")):
                os.add_dll_directory(str(d))
                os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
                _added.append(str(d))
    return _added
