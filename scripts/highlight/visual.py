"""Stage B visual analysis on shortlisted windows only.

Frames are sampled at a low rate (default 2 fps, 256 px wide, grayscale) with one ffmpeg call per
window; NVIDIA decode (-hwaccel cuda) is used when available and falls back to CPU decode on any error.
A video-wide baseline comes from a few short random windows, so motion is judged relative to this
stream (a racing stream always moves a lot; a menu-heavy stream rarely does).

Per window: gameplay motion, cut/flash events, facecam motion (inside the cam box), and a 64-bit
difference hash of three frames for duplicate detection. No frame is sent anywhere.
"""
import shutil, subprocess
import numpy as np
from . import accel

WIDTH = 256
ZCAP = 20.0  # report at most 20 sigma; the baseline scale has a floor, so static scenes give huge z


def _dims(video):
    exe = shutil.which("ffprobe") or "ffprobe"
    r = subprocess.run([exe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0:s=x", str(video)], capture_output=True, text=True)
    try:
        w, h = map(int, r.stdout.strip().split("\n")[0].split("x")[:2])
    except ValueError:
        return None
    hh = int(round(h * WIDTH / w / 2) * 2)
    return WIDTH, max(2, hh)


def frames(video, start, dur, fps=2.0, hwaccel="auto"):
    """Gray uint8 array [n, h, w] sampled at `fps` over [start, start+dur). Empty array if unreadable."""
    dims = _dims(video)
    if not dims or dur <= 0:
        return np.zeros((0, 1, 1), np.uint8)
    W, H = dims
    hw = accel.decode_hwaccel(hwaccel)
    for attempt in ([hw] if hw else []) + [None]:
        cmd = [shutil.which("ffmpeg") or "ffmpeg", "-v", "error"]
        if attempt:
            cmd += ["-hwaccel", attempt]
        cmd += ["-ss", f"{max(0.0, start):.3f}", "-t", f"{dur:.3f}", "-i", str(video), "-an",
                "-vf", f"fps={fps},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0 and len(r.stdout) >= W * H:
            n = len(r.stdout) // (W * H)
            return np.frombuffer(r.stdout[: n * W * H], np.uint8).reshape(n, H, W)
    return np.zeros((0, H, W), np.uint8)


def _region(fr, box):
    if box is None:
        return None
    n, H, W = fr.shape
    x, y, w, h = box
    x0, y0, x1, y1 = int(x * W), int(y * H), int((x + w) * W), int((y + h) * H)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return slice(y0, y1), slice(x0, x1)


def motion(fr, box=None):
    """(game_motion[n-1], cam_motion[n-1] or None, brightness[n]) per frame step."""
    if len(fr) < 2:
        return np.zeros(0), None, fr.reshape(len(fr), -1).mean(1) if len(fr) else np.zeros(0)
    f = fr.astype(np.int16)
    d = np.abs(np.diff(f, axis=0))
    reg = _region(fr, box)
    mask = np.ones(fr.shape[1:], bool)
    cam = None
    if reg:
        mask[reg] = False
        cam = d[:, reg[0], reg[1]].reshape(len(d), -1).mean(1)
    game = d[:, mask].mean(1)
    bright = f[:, mask].mean(1)
    return game, cam, bright


def dhash(img):
    """64-bit difference hash of a grayscale frame."""
    h, w = img.shape
    ys = np.linspace(0, h - 1, 8).astype(int)
    xs = np.linspace(0, w - 1, 9).astype(int)
    g = img[np.ix_(ys, xs)].astype(int)
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(int(a) ^ int(b)).count("1")


def baseline(video, duration, avoid, box=None, n=8, win=3.0, fps=2.0, seed=0, hwaccel="auto"):
    """Median/MAD of gameplay and cam motion over random windows away from candidate windows."""
    rng = np.random.default_rng(seed)
    gs, cs = [], []
    tries = 0
    while len(gs) < n * 4 and tries < n * 4 and duration > win * 2:
        tries += 1
        t = float(rng.uniform(0, duration - win))
        if any(t < e and t + win > s for s, e in avoid):
            continue
        g, c, _ = motion(frames(video, t, win, fps, hwaccel), box)
        gs += list(g)
        if c is not None:
            cs += list(c)
        if tries >= n and len(gs) >= n:
            break

    def stats(v):
        v = np.asarray(v, float)
        if len(v) < 3:
            return None
        med = float(np.median(v))
        return {"median": med, "scale": float(max(1.4826 * np.median(np.abs(v - med)), 0.5))}
    return {"game": stats(gs), "cam": stats(cs)}


def analyse(video, start, end, ev_start, ev_end, base, box=None, fps=2.0, hwaccel="auto"):
    fr = frames(video, start, end - start, fps, hwaccel)
    out = {"frames": int(len(fr)), "game_motion_z": 0.0, "cam_motion_z": None, "cuts": 0, "flash": 0.0,
           "event_offset": None, "hashes": []}
    if len(fr) < 2:
        return out
    game, cam, bright = motion(fr, box)
    ts = start + (np.arange(len(game)) + 1) / fps
    ev = (ts >= ev_start - 1) & (ts <= ev_end + 1)
    if not ev.any():
        ev[:] = True
    gb = base.get("game") if base else None
    if gb:
        gz = (game - gb["median"]) / gb["scale"]
        out["game_motion_z"] = round(min(ZCAP, float(gz[ev].max())), 2)
        big = gz > 6
        out["cuts"] = int(big.sum())
        hit = np.where(gz > 3)[0]
        if len(hit):
            out["event_offset"] = round(float(ts[hit[0]] - start), 2)
    cb = base.get("cam") if base else None
    if cam is not None and cb:
        cz = (cam - cb["median"]) / cb["scale"]
        out["cam_motion_z"] = round(min(ZCAP, float(cz[ev].max())), 2)
        hit = np.where(cz > 3)[0]
        if len(hit) and (out["event_offset"] is None or ts[hit[0]] - start < out["event_offset"]):
            out["event_offset"] = round(float(ts[hit[0]] - start), 2)
    jumps = np.diff(bright)
    out["flash"] = round(float(max(0.0, jumps.max())) if len(jumps) else 0.0, 1)
    k = len(fr)
    out["hashes"] = [dhash(fr[int(k * q)]) for q in (0.25, 0.5, 0.75)]
    return out
