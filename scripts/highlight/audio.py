"""Stage A audio features, one value per second, computed in 60 s chunks (bounded memory).

- db:          RMS level (dBFS)
- jump:        level above the local 30 s baseline (dB)
- onset:       sharpest 300 ms rise in short-term level inside the second (dB) -> surprise
- bright:      energy share above 1.5 kHz while loud -> shouting / excited voice timbre
- laugh:       0..1 laughter-likeness: regular 3.5-7.5 Hz bursts with deep level modulation
- active:      fraction of 10 ms frames above the noise floor (dead-air detection)

The laughter detector is a signal-processing heuristic, not a trained classifier. It responds to
rhythmic "ha-ha-ha" bursts and can also fire on rhythmic gameplay sounds; Stage B cross-checks it.
"""
import wave
import numpy as np
from .util import smooth

SR = 16000
HOP = 160          # 10 ms
WIN = 512          # 32 ms analysis window
CHUNK_S = 60


def read_wav(path):
    with wave.open(str(path)) as w:
        if w.getframerate() != SR or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16 kHz mono 16-bit WAV (as written by moments.audio_rms)")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def _frames(x):
    n = 1 + max(0, (len(x) - WIN) // HOP)
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx] if len(x) >= WIN else np.zeros((0, WIN), np.float32)


def frame_features(x):
    """Per 10 ms frame: level dB and high-band energy share."""
    win = np.hanning(WIN).astype(np.float32)
    freqs = np.fft.rfftfreq(WIN, 1 / SR)
    hi = freqs >= 1500
    sp = freqs >= 80
    dbs, his = [], []
    step = CHUNK_S * SR
    for c0 in range(0, len(x), step):
        seg = x[c0: c0 + step + WIN]
        fr = _frames(seg)
        fr = fr[: (min(step, len(x) - c0)) // HOP]
        if len(fr) == 0:
            continue
        rms = np.sqrt((fr ** 2).mean(1) + 1e-10)
        p = np.abs(np.fft.rfft(fr * win, axis=1)) ** 2
        tot = p[:, sp].sum(1) + 1e-12
        dbs.append(20 * np.log10(rms))
        his.append(p[:, hi].sum(1) / tot)
    if not dbs:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(dbs), np.concatenate(his)


def _laugh_frames(fdb, active):
    """Laughter-likeness per second from the 100 Hz level envelope over a 2 s window."""
    n = len(fdb) // 100
    out = np.zeros(n)
    env = np.power(10, fdb / 20)  # linear amplitude
    lags = np.arange(13, 29)      # 130-280 ms period = 3.6-7.7 Hz
    for s in range(n):
        a, b = max(0, s * 100 - 50), min(len(env), s * 100 + 150)
        e = env[a:b]
        if len(e) < 120 or active[s] < 0.35:
            continue
        # high-pass the envelope (remove anything slower than ~400 ms) so a single level step - a sound
        # starting or stopping inside the window - does not look like deep rhythmic modulation
        ec = e - np.convolve(np.pad(e, 20, mode="edge"), np.ones(41) / 41, mode="valid")
        depth = ec.std() / (e.mean() + 1e-9)
        den = (ec ** 2).sum() + 1e-12
        acf = np.array([(ec[:-L] * ec[L:]).sum() / den for L in lags])
        reg = acf.max()
        # regular bursts (reg) with deep modulation (depth); speech is less periodic
        out[s] = float(np.clip((reg - 0.35) / 0.40, 0, 1) * np.clip((depth - 0.45) / 0.45, 0, 1))
    return out


def suppress_sustained(laugh, thr=0.3, max_run=8, factor=0.2):
    """Laughter bouts are short. A rhythm that keeps going for > max_run s is music, an engine or a
    siren: scale it down instead of reporting a 20 s laugh."""
    out = laugh.copy()
    on = laugh > thr
    i = 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and on[j]:
                j += 1
            if j - i > max_run:
                out[i:j] *= factor
            i = j
        else:
            i += 1
    return out


def features(wav_path=None, x=None):
    """Dict of per-second numpy arrays (all the same length)."""
    if x is None:
        x = read_wav(wav_path)
    n = len(x) // SR
    if n == 0:
        z = np.zeros(0)
        return {k: z for k in ("db", "jump", "onset", "bright", "laugh", "active")}
    sec = x[: n * SR].reshape(n, SR)
    db = 20 * np.log10(np.sqrt((sec ** 2).mean(1) + 1e-9))
    fdb, fhi = frame_features(x[: n * SR])
    fdb, fhi = fdb[: n * 100], fhi[: n * 100]
    if len(fdb) < n * 100:  # pad the tail frame
        fdb = np.pad(fdb, (0, n * 100 - len(fdb)), mode="edge")
        fhi = np.pad(fhi, (0, n * 100 - len(fhi)), mode="edge")
    floor = np.percentile(fdb, 20) + 6.0
    act_f = fdb > max(floor, -60.0)
    active = act_f.reshape(n, 100).mean(1)
    st = smooth(fdb, 10)                       # 100 ms short-term level
    rise = np.zeros_like(st)
    rise[30:] = st[30:] - st[:-30]             # rise over 300 ms
    onset = rise.reshape(n, 100).max(1)
    w = np.where(act_f, np.power(10, fdb / 10), 0).reshape(n, 100)
    bright = (fhi.reshape(n, 100) * w).sum(1) / (w.sum(1) + 1e-12)
    jump = db - smooth(db, 30)
    laugh = suppress_sustained(_laugh_frames(fdb, active))
    return {"db": db, "jump": jump, "onset": np.clip(onset, 0, None), "bright": bright, "laugh": laugh, "active": active}
