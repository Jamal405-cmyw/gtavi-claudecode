"""Facecam-aware vertical reframing for cut.py (no new dependencies: numpy + ffmpeg via highlight.visual).

Gameplay panel (techniques adapted, no code copied):
  - ClipsAI: one *static* crop per segment, segments split at scene changes, identical neighbours
    merged -> no jittery panning.
  - PySceneDetect AdaptiveDetector: a cut is a frame whose content delta is >= 3x the mean of its
    neighbours and >= 15 (luma, 0-255), with a minimum scene length.
  Crop position per segment = argmax of  centre prior (GTA's third-person character sits near the
  middle)  +  a small motion-saliency nudge  -  a large penalty for including the webcam overlay
  (otherwise the streamer's face appears twice). The nudge is capped (max_shift) because in a
  third-person game the whole world moves; motion alone would drag the crop away from the player.

Facecam panel: crop the webcam box to the panel's aspect ratio *around the detected face*
(face centred horizontally, eyes near the upper third) instead of around the box centre.
"""
import numpy as np


def crop_width_frac(src_w, src_h, panel_w, panel_h):
    """Fraction of the source width that a panel (scaled to panel_h, cropped to panel_w) shows."""
    return float(min(1.0, (panel_w / panel_h) / (src_w / src_h)))


def scene_cuts(fr, fps, ratio=3.0, min_delta=15.0, window=2, min_len=1.0):
    """Cut times (s, relative to the first frame) using the AdaptiveDetector rule on luma deltas."""
    if len(fr) < 3:
        return []
    d = np.abs(np.diff(fr.astype(np.int16), axis=0)).reshape(len(fr) - 1, -1).mean(1)
    cuts, last = [], -1e9
    for i in range(len(d)):
        nb = np.r_[d[max(0, i - window):i], d[i + 1:i + 1 + window]]
        base = nb.mean() if len(nb) else 0.0
        if d[i] >= min_delta and d[i] >= ratio * max(base, 1e-6):
            t = (i + 1) / fps
            if t - last >= min_len:
                cuts.append(round(t, 3)); last = t
    return cuts


def _best_center(sal_cols, cw, cam=None, max_shift=0.15, prior_sigma=0.22, motion_weight=0.6):
    """Crop centre (fraction of width) maximising centre prior + capped motion nudge - webcam overlap."""
    if cw >= 0.999:
        return 0.5
    n = len(sal_cols)
    xs = np.linspace(cw / 2, 1 - cw / 2, 201)
    cum = np.r_[0.0, np.cumsum(sal_cols)]
    tot = cum[-1] if cum[-1] > 0 else 1.0
    best, bx = -1e18, 0.5
    for c in xs:
        a, b = int(round((c - cw / 2) * n)), int(round((c + cw / 2) * n))
        motion = (cum[b] - cum[a]) / tot                         # share of motion inside the window
        prior = np.exp(-0.5 * ((c - 0.5) / prior_sigma) ** 2)
        pen = 0.0
        if cam is not None:
            ov = max(0.0, min(c + cw / 2, cam[0] + cam[2]) - max(c - cw / 2, cam[0]))
            pen = 10.0 * ov / max(cam[2], 1e-6)                  # any webcam inside the crop is costly
        score = prior + motion_weight * motion - pen
        if cam is None and abs(c - 0.5) > max_shift:
            score -= 5.0                                          # cap the motion nudge
        if score > best:
            best, bx = score, c
    return float(bx)


def plan_game_crop(frames, fps, cw, cam=None, min_seg=1.5, merge_tol=0.04):
    """Piecewise-static crop plan: [{t0, t1, cx}] (seconds relative to clip start, cx = centre fraction).
    frames: gray uint8 [n, h, w] sampled at `fps` from the clip; cam: webcam box (x, y, w, h) fractions."""
    n = len(frames)
    dur = n / fps if n else 0.0
    if n < 2:
        return {"segments": [{"t0": 0.0, "t1": dur, "cx": _best_center(np.ones(8), cw, cam)}], "cuts": []}
    cuts = scene_cuts(frames, fps)
    edges = [0.0] + cuts + [dur]
    segs = []
    motion = np.abs(np.diff(frames.astype(np.int16), axis=0))  # motion[i] ~ between frame i and i+1
    for t0, t1 in zip(edges[:-1], edges[1:]):
        i0, i1 = int(round(t0 * fps)), max(int(round(t0 * fps)) + 1, int(round(t1 * fps)) - 1)
        m = motion[i0:i1]                                   # never across the cut itself
        cols = m.mean(axis=(0, 1)) if len(m) else np.ones(frames.shape[2])
        if cam is not None:                                 # webcam motion is not gameplay
            W = frames.shape[2]
            cols = cols.astype(float).copy(); cols[int(cam[0] * W):int((cam[0] + cam[2]) * W) + 1] *= 0.0
        segs.append({"t0": round(t0, 3), "t1": round(t1, 3), "cx": round(_best_center(cols, cw, cam), 4)})
    # merge short segments into the previous one, then identical neighbours (ClipsAI _merge_identical_segments)
    out = []
    for s in segs:
        if out and (s["t1"] - s["t0"] < min_seg or abs(s["cx"] - out[-1]["cx"]) <= merge_tol):
            out[-1]["t1"] = s["t1"]
        else:
            out.append(dict(s))
    if len(out) > 1 and out[0]["t1"] - out[0]["t0"] < min_seg:
        out[1]["t0"] = out[0]["t0"]; out.pop(0)
    return {"segments": out, "cuts": cuts}


def crop_x_expr(segments):
    """ffmpeg crop `x` expression (scaled-input pixels) for a piecewise-static plan."""
    expr = f"{segments[-1]['cx']:.4f}"
    for s in reversed(segments[:-1]):
        expr = f"if(lt(t\\,{s['t1']:.3f})\\,{s['cx']:.4f}\\,{expr})"
    return f"clip(iw*{expr}-ow/2\\,0\\,iw-ow)"


def face_crop(cam, face, src_w, src_h, panel_w, panel_h, eye_line=0.40):
    """Sub-rectangle of the webcam box with the panel's aspect ratio, centred on the face.
    cam/face: (x, y, w, h) fractions of the frame. Returns (x, y, w, h) fractions."""
    cx, cy, cwf, chf = cam
    bw, bh = cwf * src_w, chf * src_h
    A = panel_w / panel_h
    if bw / bh > A:
        w, h = bh * A, bh
    else:
        w, h = bw, bw / A
    fx = (face[0] + face[2] / 2) * src_w
    fy = (face[1] + face[3] / 2) * src_h
    x0 = min(max(fx - w / 2, cx * src_w), cx * src_w + bw - w)
    y0 = min(max(fy - eye_line * h, cy * src_h), cy * src_h + bh - h)
    return (x0 / src_w, y0 / src_h, w / src_w, h / src_h)


def plan_for_clip(video, s0, e0, src_w, src_h, panel_w, panel_h, cam=None, fps=4.0, hwaccel="auto"):
    """Sample the clip and return the plan (+ the crop width) for the gameplay panel."""
    from highlight import visual
    fr = visual.frames(video, s0, e0 - s0, fps=fps, hwaccel=hwaccel)
    cw = crop_width_frac(src_w, src_h, panel_w, panel_h)
    p = plan_game_crop(fr, fps, cw, cam)
    p.update(crop_w=round(cw, 4), frames=int(len(fr)), fps=fps)
    return p
