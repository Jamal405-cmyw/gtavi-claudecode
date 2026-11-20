"""Speech segments for clip boundaries (Stage B, shortlisted windows only).

Primary: Silero VAD v6 as bundled inside faster-whisper (`faster_whisper.vad`, ONNX on CPU) - no extra
package and no download. Whisper word timestamps drift by a few hundred ms and say nothing when the
transcript is missing; VAD gives real speech onsets/offsets to cut on.

Fallback (faster-whisper not importable): Auto-Editor's audio method - 20 ms frames whose peak
amplitude exceeds 0.04 of full scale are "active", with 0.2 s margin, gaps < 0.2 s closed and blips
< 0.1 s dropped. It detects *sound*, not speech (game audio counts), so the result says which ran.
"""
import numpy as np

SR = 16000


def merge_windows(windows, gap=1.0):
    out = []
    for a, b in sorted((max(0.0, a), b) for a, b in windows if b > a):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(w) for w in out]


def _silero(x, min_silence_ms=250, pad_ms=100):
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    seg = get_speech_timestamps(np.ascontiguousarray(x, dtype=np.float32),
                                VadOptions(min_silence_duration_ms=min_silence_ms, speech_pad_ms=pad_ms))
    return [(s["start"] / SR, s["end"] / SR) for s in seg]


def energy_segments(x, thr=0.04, margin=0.2, min_cut=0.2, min_clip=0.1, frame=0.02):
    """Auto-Editor style loud/silent split (peak amplitude per frame vs threshold)."""
    n = int(frame * SR)
    k = len(x) // n
    if k == 0:
        return []
    peak = np.abs(x[: k * n].reshape(k, n)).max(1)
    on = peak > thr
    m = int(round(margin / frame))
    if m:
        on = np.convolve(on.astype(int), np.ones(2 * m + 1, int), mode="same") > 0
    segs, i = [], 0
    while i < k:
        if on[i]:
            j = i
            while j < k and on[j]:
                j += 1
            segs.append([i * frame, j * frame])
            i = j
        else:
            i += 1
    merged = []
    for s in segs:  # close short silences (min cut), then drop short blips (min clip)
        if merged and s[0] - merged[-1][1] < min_cut:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [(round(a, 3), round(b, 3)) for a, b in merged if b - a >= min_clip]


def segments(x, windows, method="auto"):
    """Speech (or sound) segments in absolute seconds inside the given windows. Returns (segments, method)."""
    used = "silero"
    if method == "energy":
        used = "energy"
    else:
        try:
            __import__("faster_whisper.vad")  # probe: package + onnxruntime importable
        except Exception:
            if method == "silero":
                raise
            used = "energy"
    out = []
    for a, b in merge_windows(windows):
        lo, hi = int(a * SR), min(len(x), int(b * SR))
        if hi - lo < SR // 2:
            continue
        chunk = x[lo:hi]
        segs = _silero(chunk) if used == "silero" else energy_segments(chunk)
        out += [(round(a + s, 3), round(a + e, 3)) for s, e in segs]
    return sorted(out), used
