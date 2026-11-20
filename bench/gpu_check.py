"""Verify faster-whisper on the GPU and measure it (no CPU fallback: a GPU failure is reported as such).

usage: python bench/gpu_check.py audio.wav [--reference script.txt] [--model large-v3-turbo]
                                 [--device cuda] [--compute float16] [--out result.json]

Reports: CUDA DLL folders registered, model load time, transcription time, real-time factor, GPU memory
(whole-GPU nvidia-smi samples every 0.1 s: idle baseline before loading, peak while running; the delta is
what this process added) and, with --reference, word error rate against the known text. Transcription
settings and input path match moments.transcribe (16 kHz mono samples, word timestamps, VAD filter).
"""
import argparse, json, re, shutil, subprocess, sys, threading, time, wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def gpu_mem_mb():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    r = subprocess.run([exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
    try:
        return float(r.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


class Sampler:
    def __init__(self, every=0.1):
        self.vals, self.stop, self.every = [], threading.Event(), every
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            v = gpu_mem_mb()
            if v is not None:
                self.vals.append(v)
            time.sleep(self.every)

    def __enter__(self):
        self.t.start(); return self

    def __exit__(self, *a):
        self.stop.set(); self.t.join(2)


def words_of(text):
    return re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))


def wer(ref, hyp):
    """Word error rate (Levenshtein on words)."""
    r, h = words_of(ref), words_of(hyp)
    if not r:
        return None
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return round(prev[-1] / len(r), 4)


def load_pcm(path):
    """16 kHz mono float32 samples via ffmpeg, as moments.audio_rms/transcribe do. faster-whisper 1.2.1's
    own file decoder calls av.open(metadata_errors=...), which PyAV 19 rejects, so never pass a path."""
    import numpy as np
    from common import tool
    r = subprocess.run([tool("ffmpeg"), "-v", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.int16).astype(np.float32) / 32768


def audio_seconds(path):
    try:
        with wave.open(str(path)) as w:
            return w.getnframes() / w.getframerate()
    except wave.Error:
        from common import ffprobe
        return float(ffprobe(path)["format"]["duration"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("audio"); ap.add_argument("--reference"); ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--compute", default="float16"); ap.add_argument("--out")
    a = ap.parse_args(argv)
    import cuda_dlls
    res = {"audio": str(a.audio), "audio_s": round(audio_seconds(a.audio), 2), "model": a.model, "device": a.device,
           "compute_type": a.compute, "dll_dirs": cuda_dlls.register()}
    import ctranslate2, faster_whisper
    res.update(ctranslate2=ctranslate2.__version__, faster_whisper=faster_whisper.__version__,
               cuda_devices=ctranslate2.get_cuda_device_count())
    idle = [v for v in (gpu_mem_mb() for _ in range(5)) if v is not None]
    res["gpu_mem_idle_mb"] = min(idle) if idle else None
    try:
        with Sampler() as s_load:
            t = time.perf_counter()
            model = faster_whisper.WhisperModel(a.model, device=a.device, compute_type=a.compute)
            res["load_s"] = round(time.perf_counter() - t, 2)
        pcm = load_pcm(a.audio)  # same input path as moments.transcribe (samples, not a file path)
        with Sampler() as s_run:
            t = time.perf_counter()
            segs, info = model.transcribe(pcm, word_timestamps=True, vad_filter=True)
            words = [w for seg in segs for w in (seg.words or [])]
            res["transcribe_s"] = round(time.perf_counter() - t, 2)
        res["status"] = "ok"
        res["words"] = len(words)
        res["language"] = info.language
        res["text"] = " ".join(w.word.strip() for w in words)
        res["realtime_factor"] = round(res["audio_s"] / res["transcribe_s"], 1) if res["transcribe_s"] else None
        peak = max(s_load.vals + s_run.vals) if (s_load.vals or s_run.vals) else None
        res["gpu_mem_peak_mb"] = peak
        res["gpu_mem_added_mb"] = round(peak - res["gpu_mem_idle_mb"]) if peak and res["gpu_mem_idle_mb"] is not None else None
        res["gpu_samples"] = len(s_load.vals) + len(s_run.vals)
        if a.reference:
            res["wer"] = wer(Path(a.reference).read_text(encoding="utf-8"), res["text"])
    except Exception as ex:  # reported, never silently replaced by a CPU run
        res["status"] = "failed"
        res["error"] = f"{type(ex).__name__}: {ex}"
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "text"}, indent=1))
    print("text:", (res.get("text") or "")[:400])
    return res


if __name__ == "__main__":
    main()
