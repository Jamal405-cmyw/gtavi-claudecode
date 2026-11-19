"""Small numeric helpers shared by the stages. numpy only."""
import numpy as np


def smooth(a, k):
    """Moving average with edge reflection (no zero-padding bias at the ends)."""
    a = np.asarray(a, dtype=float)
    k = int(max(1, k))
    if k == 1 or len(a) == 0:
        return a.copy()
    pad = k // 2
    p = np.pad(a, (pad, k - 1 - pad), mode="reflect" if len(a) > pad else "edge")
    return np.convolve(p, np.ones(k) / k, mode="valid")


def robust_z(a):
    """(x - median) / (1.4826 * MAD); falls back to std, then zeros for flat input."""
    a = np.asarray(a, dtype=float)
    if len(a) == 0:
        return a
    med = np.median(a)
    mad = 1.4826 * np.median(np.abs(a - med))
    scale = mad if mad > 1e-6 else a.std()
    if scale < 1e-9:
        return np.zeros_like(a)
    return (a - med) / scale


def pos(a, cap=4.0):
    return np.clip(np.asarray(a, dtype=float), 0.0, cap)


def shift_earlier(a, sec):
    """Move a per-second series `sec` seconds earlier (chat reacts after the event)."""
    a = np.asarray(a, dtype=float)
    sec = int(round(sec))
    if sec <= 0 or len(a) == 0:
        return a.copy()
    out = np.zeros_like(a)
    out[: max(0, len(a) - sec)] = a[sec:]
    return out


def sat(x):
    """Saturating map [0, inf) -> [0, 1)."""
    return float(1.0 - np.exp(-max(0.0, float(x))))


def peaks(score, k, min_sep):
    """Indices of up to k local maxima > 0, greedy non-maximum suppression with min_sep seconds."""
    score = np.asarray(score, dtype=float)
    taken = []
    for i in np.argsort(-score, kind="stable"):
        if len(taken) >= k or score[i] <= 0:
            break
        if all(abs(int(i) - j) >= min_sep for j in taken):
            taken.append(int(i))
    return taken


def window_max(a, s, e):
    a = np.asarray(a)
    s, e = max(0, int(s)), min(len(a), int(np.ceil(e)))
    return float(a[s:e].max()) if e > s else 0.0


def window_mean(a, s, e):
    a = np.asarray(a)
    s, e = max(0, int(s)), min(len(a), int(np.ceil(e)))
    return float(a[s:e].mean()) if e > s else 0.0
