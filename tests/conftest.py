import shutil, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "bench")]

needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg not installed")


def W(text, t0=0.0, step=0.4):
    """Word list helper: evenly spaced words; '|' inserts a 1 s pause."""
    out, t = [], t0
    for tok in text.split():
        if tok == "|":
            t += 1.0; continue
        out.append({"w": tok, "s": round(t, 2), "e": round(t + step * 0.8, 2)})
        t += step
    return out
