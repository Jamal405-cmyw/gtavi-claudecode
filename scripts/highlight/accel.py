"""GPU / CPU selection. Everything degrades to CPU; nothing here is required to succeed."""
import functools, shutil, subprocess


@functools.lru_cache(maxsize=1)
def cuda_devices():
    try:
        import cuda_dlls
        cuda_dlls.register()
    except ImportError:
        pass
    try:
        import ctranslate2
        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


@functools.lru_cache(maxsize=1)
def ffmpeg_hwaccels():
    exe = shutil.which("ffmpeg")
    if not exe:
        return ()
    try:
        r = subprocess.run([exe, "-hide_banner", "-hwaccels"], capture_output=True, text=True, timeout=15)
        lines = [ln.strip() for ln in r.stdout.splitlines()[1:] if ln.strip()]
        return tuple(lines)
    except Exception:
        return ()


def decode_hwaccel(pref="auto"):
    """ffmpeg -hwaccel value for frame sampling, or None for software decode."""
    if pref in (None, "", "none", "cpu"):
        return None
    acc = ffmpeg_hwaccels()
    if pref != "auto":
        return pref if pref in acc else None
    if cuda_devices() > 0 and "cuda" in acc:
        return "cuda"
    return None


def whisper_choice(model="auto", device="auto"):
    """(model, device, compute_type). auto: large-v3-turbo fp16 on an NVIDIA GPU, small int8 on CPU."""
    dev = device
    if dev == "auto":
        dev = "cuda" if cuda_devices() > 0 else "cpu"
    mdl = model
    if mdl == "auto":
        mdl = "large-v3-turbo" if dev == "cuda" else "small"
    return mdl, dev, ("float16" if dev == "cuda" else "int8")


def summary():
    return {"cuda_devices": cuda_devices(), "ffmpeg_hwaccels": list(ffmpeg_hwaccels()),
            "decode_hwaccel": decode_hwaccel("auto"), "whisper": list(whisper_choice())}
