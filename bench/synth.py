"""Deterministic synthetic stream material for tests and the synthetic benchmark suite.

Nothing here is real footage. The generators produce signals with the *acoustic shape* of stream
events (irregular speech syllables, rhythmic laughter, bright loud shouting, explosions) and of
common false positives (a loud music bed, loud gameplay nobody reacts to, a mic bump, a chat spam bot). They exist so the pipeline
can be exercised end-to-end and reproducibly without downloading anyone's stream. Results on this
suite say the mechanics work; they do NOT measure real-world clip quality (see bench/README.md).
"""
import json, subprocess, wave
from pathlib import Path
import numpy as np

SR = 16000


def _harm(t, f0, gains):
    out = np.zeros_like(t)
    for k, g in enumerate(gains, 1):
        if k * f0 < SR / 2:
            out += g * np.sin(2 * np.pi * k * f0 * t)
    return out


def _formant_gains(f0, f1, f2, k_max=30, bright=0.0):
    ks = np.arange(1, k_max + 1) * f0
    g = np.exp(-((ks - f1) / 250) ** 2) + 0.6 * np.exp(-((ks - f2) / 350) ** 2) + 0.15 / np.arange(1, k_max + 1)
    g *= 1 + bright * (ks / 2000.0)
    return g / g.max()


def speech(dur, rng, level=0.08, f0=(110, 170), bright=0.0):
    """Irregular syllables (110-320 ms) with short pauses; pitch and formants vary per syllable."""
    out = np.zeros(int(dur * SR))
    pos = 0
    while pos < len(out):
        syl = int(rng.uniform(0.11, 0.32) * SR)
        t = np.arange(syl) / SR
        f = rng.uniform(*f0)
        g = _formant_gains(f, rng.uniform(450, 800), rng.uniform(1100, 2200), bright=bright)
        env = np.sin(np.pi * np.linspace(0, 1, syl)) ** 0.7
        s = _harm(t, f, g) * env * rng.uniform(0.6, 1.0)
        s += rng.normal(0, 0.05, syl) * env
        end = min(len(out), pos + syl)
        out[pos:end] += s[: end - pos]
        pos = end + int(rng.uniform(0.02, 0.09) * SR)
        if rng.random() < 0.12:
            pos += int(rng.uniform(0.25, 0.7) * SR)  # phrase pause
    return level * out / (np.abs(out).max() + 1e-9)


def laugh(dur, rng, level=0.22, rate=5.0):
    """Regular 'ha' bursts at ~rate Hz with breathy noise - the rhythm the laughter heuristic keys on."""
    out = np.zeros(int(dur * SR))
    period = int(SR / rate)
    on = int(period * 0.55)
    f = rng.uniform(230, 290)
    g = _formant_gains(f, 750, 1300, bright=0.5)
    for p in range(0, len(out) - on, period):
        t = np.arange(on) / SR
        env = np.sin(np.pi * np.linspace(0, 1, on)) ** 0.5
        s = (_harm(t, f * rng.uniform(0.95, 1.05), g) + rng.normal(0, 0.35, on)) * env
        out[p:p + on] += s
    return level * out / (np.abs(out).max() + 1e-9)


def shout(dur, rng, level=0.35):
    return speech(dur, rng, level=level, f0=(260, 360), bright=1.5)


def explosion(dur, rng, level=0.5):
    n = int(dur * SR)
    x = np.cumsum(rng.normal(0, 1, n)); x -= np.convolve(x, np.ones(400) / 400, mode="same")
    x = x / (np.abs(x).max() + 1e-9) + 0.3 * rng.normal(0, 1, n)
    return level * x * np.exp(-np.arange(n) / (0.5 * SR)) / 1.3


def music(dur, rng, level=0.3):
    t = np.arange(int(dur * SR)) / SR
    x = sum(np.sin(2 * np.pi * f * t) for f in (220, 277.2, 329.6, 440))
    beat = 0.6 + 0.4 * (np.sin(2 * np.pi * 2 * t) > 0)
    return level * x / 4 * beat


def click(rng, level=0.9):
    n = int(0.05 * SR)
    return level * rng.normal(0, 0.5, n) * np.exp(-np.arange(n) / (0.01 * SR))


def bed(dur, rng, level=0.02):
    """Quiet gameplay ambience."""
    n = int(dur * SR)
    x = np.convolve(rng.normal(0, 1, n), np.ones(30) / 30, mode="same")
    return level * x / (np.abs(x).max() + 1e-9)


def add(buf, sig, t):
    a = int(t * SR); b = min(len(buf), a + len(sig))
    if b > a:
        buf[a:b] += sig[: b - a]


def write_wav(path, x):
    y = np.clip(x, -1, 1)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((y * 32767).astype(np.int16).tobytes())


# ---------- full labelled episodes ----------

FILLER = ("so I think we should go over there and check the shop first then grab a car maybe the one by "
          "pier because it was faster last time okay let me see map again where is mission marker this road "
          "goes north probably need more money for upgrades chat what do you think about blue paint kind of "
          "slow today honestly traffic everywhere bridge tunnel garage beach hotel airport quest side story").split()


def words_for(t0, t1, rng, vocab=FILLER):
    """Plain filler transcript words between t0 and t1 (about 2.5 words/s)."""
    out, t = [], t0
    while t < t1 - 0.3:
        d = rng.uniform(0.18, 0.42)
        out.append({"w": str(vocab[int(rng.integers(0, len(vocab)))]), "s": round(t, 2), "e": round(t + d, 2)})
        t += d + rng.uniform(0.02, 0.12)
        if rng.random() < 0.1:
            out[-1]["w"] += "."
            t += rng.uniform(0.6, 1.1)
    return out


def phrase_words(t0, text, step=0.3):
    return [{"w": w, "s": round(t0 + i * step, 2), "e": round(t0 + i * step + step * 0.8, 2)} for i, w in enumerate(text.split())]


def episode(seed, dur=240, n_events=3, n_decoys=2):
    """One labelled synthetic stream. Returns (audio, words, chat_msgs, visual_events, labels)."""
    rng = np.random.default_rng(seed)
    x = bed(dur, rng)
    words, chat, vis, labels = [], [], [], []
    # chat baseline: ~1 msg/s from a pool of viewers
    for t in np.cumsum(rng.exponential(1.0, int(dur * 1.3))):
        if t < dur:
            chat.append((float(t), f"viewer{rng.integers(0, 400)}", str(rng.choice(["hi", "nice car", "where is that", "lol", "go left", "first time?"]))))
    slots = np.linspace(25, dur - 30, n_events + n_decoys)
    kinds = (["laugh", "chuckle", "shout", "explosion"] * 3)[:n_events] + (["gunfight", "music", "click", "spam"] * 3)[:n_decoys]
    order = rng.permutation(len(slots))
    talk_ranges = []
    for slot_i, k in zip(order, kinds):
        t = float(slots[slot_i] + rng.uniform(-4, 4))
        if k == "laugh":
            add(x, speech(6, rng), t - 6); words += phrase_words(t - 5, "wait watch this I am going to jump the bridge", 0.45)
            add(x, laugh(3.5, rng), t); words += phrase_words(t, "hahaha no way", 0.6)
            vis.append(("cam", t, 3.5))
            chat += [(t + 3 + rng.uniform(0, 6), f"fan{i}", "KEKW") for i in range(40)]
            labels.append({"start": t - 6, "end": t + 5, "peak": t, "kind": k})
            talk_ranges.append((t - 6, t + 5))
        elif k == "shout":
            add(x, speech(5, rng), t - 5); words += phrase_words(t - 5, "okay the cops are right behind me", 0.55)
            add(x, shout(3, rng), t); words += phrase_words(t, "oh my god let's go let's go", 0.4)
            vis.append(("game", t - 0.5, 3)); vis.append(("cam", t, 3))
            chat += [(t + 3 + rng.uniform(0, 6), f"fan{i}", "POGGERS W") for i in range(35)]
            labels.append({"start": t - 5, "end": t + 4.5, "peak": t, "kind": k})
            talk_ranges.append((t - 5, t + 4.5))
        elif k == "explosion":
            add(x, explosion(2.5, rng), t); add(x, shout(2, rng, level=0.25), t + 0.4)
            words += phrase_words(t + 0.4, "what the", 0.4)
            vis.append(("flash", t, 1.5)); vis.append(("game", t, 2.5))
            chat += [(t + 3 + rng.uniform(0, 6), f"fan{i}", "WHAT ??") for i in range(30)]
            labels.append({"start": t - 3, "end": t + 4, "peak": t, "kind": k})
            talk_ranges.append((t - 3, t + 4))
        elif k == "chuckle":  # quiet laugh at talking level: no loudness spike, chat laughs
            add(x, speech(6, rng, level=0.05), t - 6); words += phrase_words(t - 5, "did he just drive into the lake", 0.5)
            add(x, laugh(3, rng, level=0.06), t); words += phrase_words(t, "haha bro", 0.7)
            vis.append(("cam", t, 2))
            chat += [(t + 3 + rng.uniform(0, 7), f"fan{i}", "LUL") for i in range(22)]
            labels.append({"start": t - 6, "end": t + 4, "peak": t, "kind": k})
            talk_ranges.append((t - 6, t + 4))
        elif k == "gunfight":  # decoy: 15 s of loud gameplay shooting, no reaction from streamer or chat
            for j in range(12):
                add(x, explosion(0.6, rng, level=0.45), t + j * 1.25 + rng.uniform(0, 0.3))
            vis.append(("game", t, 15))
            talk_ranges.append((t, t + 15))
        elif k == "music":  # decoy: loud music bed for 20 s, nobody reacts
            add(x, music(20, rng), t)
            talk_ranges.append((t, t + 20))
        elif k == "click":  # decoy: mic bump
            add(x, click(rng), t)
        elif k == "spam":  # decoy: one bot floods chat with a link-free copypasta
            chat += [(t + i * 0.2, "spambot", "FOLLOW FOR FOLLOW") for i in range(60)]
    # filler talk everywhere else
    t = 2.0
    for a, b in sorted(talk_ranges) + [(dur, dur)]:
        if a - t > 3:
            add(x, speech(a - t - 1, rng, level=0.05), t)
            words += words_for(t, a - 1, rng)
        t = max(t, b + 1)
    words.sort(key=lambda w: w["s"])
    chat.sort()
    labels.sort(key=lambda l: l["start"])
    return x, words, chat, vis, labels


def render_video(path, dur, vis, cam_box=(0.7, 0.05, 0.28, 0.3), size=(640, 360), fps=15, audio_wav=None):
    """Gameplay-like video: slowly drifting texture; 'game' events = fast motion, 'flash' = bright frame,
    'cam' events = motion inside the facecam box. Raw frames piped to ffmpeg (libx264)."""
    W, H = size
    rng = np.random.default_rng(7)
    import cv2
    small = (rng.random((H // 12, W // 12)) * 255).astype(np.uint8)
    tex = cv2.resize(small, (W * 2, H * 2), interpolation=cv2.INTER_CUBIC)  # smooth, game-like scenery
    cx, cy, cw, ch = int(cam_box[0] * W), int(cam_box[1] * H), int(cam_box[2] * W), int(cam_box[3] * H)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-"]
    if audio_wav:
        cmd += ["-i", str(audio_wav), "-c:a", "aac", "-b:a", "96k"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p", "-shortest", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    face = (rng.random((ch, cw)) * 60 + 100).astype(np.uint8)
    for i in range(int(dur * fps)):
        t = i / fps
        speed = 1.0
        bright, cam_move = 0, 0.0
        for kind, t0, d in vis:
            if t0 <= t < t0 + d:
                if kind == "game": speed = 14.0
                if kind == "flash": bright = 120
                if kind == "cam": cam_move = 1.0
        ox = int((t * 6 * speed) % W); oy = int((t * 3 * speed) % H)
        f = tex[oy:oy + H, ox:ox + W].astype(np.int16) + bright
        fc = np.roll(face, int(cam_move * 12 * np.sin(t * 9)), axis=1).astype(np.int16)
        if cam_move:
            fc = fc + int(30 * np.sin(t * 13))
        f[cy:cy + ch, cx:cx + cw] = fc
        f[cy - 2:cy, cx:cx + cw] = 255; f[cy + ch:cy + ch + 2, cx:cx + cw] = 255
        p.stdin.write(np.clip(f, 0, 255).astype(np.uint8).tobytes())
    p.stdin.close(); p.wait()
    if p.returncode != 0:
        raise RuntimeError("ffmpeg failed rendering synthetic video")


def write_twitchdownloader(path, msgs):
    js = {"comments": [{"content_offset_seconds": round(t, 3), "commenter": {"display_name": u}, "message": {"body": x}} for t, u, x in msgs]}
    Path(path).write_text(json.dumps(js), encoding="utf-8")


def build_case(out_dir, seed, dur=240, n_events=3, n_decoys=2, video=True):
    """Writes source.mp4 (or only audio.wav), transcript.json, chat.json, labels.json into out_dir."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    x, words, chat, vis, labels = episode(seed, dur, n_events, n_decoys)
    write_wav(out / "audio_src.wav", x)
    (out / "transcript.json").write_text(json.dumps(words), encoding="utf-8")
    write_twitchdownloader(out / "chat.json", chat)
    (out / "labels.json").write_text(json.dumps({"video": "source.mp4", "duration": dur, "seed": seed,
                                                  "cam": [0.7, 0.05, 0.28, 0.3], "highlights": labels}, indent=1), encoding="utf-8")
    if video:
        render_video(out / "source.mp4", dur, vis, audio_wav=out / "audio_src.wav")
    return out
