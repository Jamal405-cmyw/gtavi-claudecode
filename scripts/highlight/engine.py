"""Orchestrates Stage A and Stage B and writes the job outputs.

Outputs in the job folder:
  highlights.json   full result: scores, confidence, evidence, boundaries, timings, device info
  highlights.md     human-readable ranking
  candidates.json   same shape the original moments.py wrote, so cut.py / translate.py keep working
  candidates.md
"""
import json, math, time
from dataclasses import dataclass, asdict, field
from pathlib import Path
import numpy as np
from . import ENGINE_VERSION, accel, audio as A, boundaries as B, candidates as C, chat as CH, dedupe as D
from . import scoring as S, text as T, visual as V


@dataclass
class Config:
    top: int = 8
    shortlist: int = 0            # 0 = max(15, 3 * top)
    min_len: float = 15.0
    max_len: float = 40.0
    min_sep: int = 20
    chat_lag: float = 5.0
    chat_offset: float = None     # unix time of video second 0, for wall-clock chat logs
    visual: bool = True
    vis_fps: float = 2.0
    cam: str = "auto"             # auto | none | x:y:w:h fractions
    hwaccel: str = "auto"
    weights: dict = field(default_factory=dict)
    seed: int = 0
    min_evidence: int = 1         # modalities that must clearly fire for a pick to be kept


def parse_cam(cam):
    try:
        v = [float(p) for p in cam.split(":")]
        return v if len(v) == 4 and all(0 <= x <= 1 for x in v) else None
    except (AttributeError, ValueError):
        return None


def _facecam(video, d, cam, notes):
    if cam == "none":
        return None
    box = parse_cam(cam)
    if box:
        return box
    cache = d / "facecam.json"
    if cache.exists():
        r = json.loads(cache.read_text(encoding="utf-8"))
    else:
        try:
            from facecam import find_cam
            r = find_cam(str(video))
        except Exception as ex:  # model download blocked, no OpenCV, unreadable video...
            notes.append(f"facecam detection unavailable ({type(ex).__name__}: {str(ex)[:80]}); facecam signal skipped")
            return None
        cache.write_text(json.dumps(r, indent=1), encoding="utf-8")
    if r.get("mode") in ("cam", "face"):
        return r["box"]
    notes.append(f"no facecam found ({r.get('reason', '')}); facecam signal skipped")
    return None


def heat_series(info, n):
    heat = np.zeros(n)
    for h in (info or {}).get("heatmap") or []:
        heat[int(h["start_time"]):int(math.ceil(h["end_time"]))] = h["value"]
    return heat


def has_video_stream(probe):
    return any(s.get("codec_type") == "video" for s in probe.get("streams", []))


def analyse(video, wav, words, duration, cfg=Config(), chat_msgs=None, info=None, d=None, has_video=True, others=()):
    """Core pipeline on already prepared inputs. Returns the result dict (also used by tests/bench)."""
    tm, notes = {}, []
    t0 = time.perf_counter()
    au = A.features(wav)
    n = len(au["db"])
    tx = T.features(words, n) if words else None
    ct = CH.features(chat_msgs, n, lag=cfg.chat_lag) if chat_msgs else None
    heat = heat_series(info, n)
    feats = {"audio": au, "text": tx, "chat": ct, "heat": heat}
    k = cfg.shortlist or max(15, 3 * cfg.top)
    st = C.run(au, tx, ct, heat, weights=cfg.weights, shortlist=k, min_sep=cfg.min_sep)
    tm["stage_a_s"] = round(time.perf_counter() - t0, 3)
    if not words:
        notes.append("no transcript: boundaries use audio only, contextual clarity is not judged")
    if not chat_msgs:
        notes.append("no chat log: chat signal skipped")

    # ---------- Stage B on the shortlist only
    t1 = time.perf_counter()
    cands = []
    top_peak = float(st.score[st.shortlist[0]]) if st.shortlist else 1.0
    for p in st.shortlist:
        b = B.compute(p, st.score, words, au["db"], duration, cfg.min_len, cfg.max_len)
        cands.append({"peak": p, "b": b})
    box, base = None, None
    do_vis = cfg.visual and has_video and video is not None and bool(cands)
    if do_vis:
        box = _facecam(video, d or Path(video).parent, cfg.cam, notes)
        base = V.baseline(video, duration, [(c["b"].start, c["b"].end) for c in cands], box, fps=cfg.vis_fps,
                          seed=cfg.seed, hwaccel=cfg.hwaccel)
    elif cfg.visual and not has_video:
        notes.append("no video stream: visual signal skipped")
    if duration < cfg.min_len:
        notes.append(f"video is {duration:.0f}s, shorter than --min {cfg.min_len:g}s: clips cannot reach the minimum length")
    tm["vis_frames"] = 0
    results = []
    for c in cands:
        b, p = c["b"], c["peak"]
        vis = V.analyse(video, b.start, b.end, b.ev_start, b.ev_end, base, box, cfg.vis_fps, cfg.hwaccel) if do_vis else None
        if vis:
            tm["vis_frames"] += vis["frames"]
        dims, conf, ev, mods = S.score(st, feats, b, words, vis, float(st.score[p]), top_peak)
        results.append({"start": b.start, "end": b.end, "peak": int(p), "length": round(b.end - b.start, 2),
                        "stage_a_score": round(float(st.score[p]), 3), "scores": dims, "confidence": conf,
                        "evidence": ev or ["overall energy above this stream's norm"], "signals": mods,
                        "text": T.text_between(words, b.start, b.end)[:600], "boundary": b.as_dict(),
                        "visual": {k: v for k, v in (vis or {}).items() if k != "hashes"} or None,
                        "hashes": (vis or {}).get("hashes", [])})
    tm["stage_b_s"] = round(time.perf_counter() - t1, 3)
    # a pick needs at least `min_evidence` modalities that clearly fire; "loudest bit of a flat video" is not one
    weak = [r for r in results if len(r["signals"]["modalities_agreeing"]) < cfg.min_evidence]
    results = [r for r in results if len(r["signals"]["modalities_agreeing"]) >= cfg.min_evidence]
    if weak:
        notes.append(f"{len(weak)} shortlisted moment(s) dropped: no signal clearly fired (min_evidence={cfg.min_evidence})")
    results.sort(key=lambda r: -r["scores"]["entertainment"])
    for i, r in enumerate(results, 1):
        r["id"] = f"{(d.name if d else 'job')}#{i}"
    kept, dropped = D.dedupe(results, others)
    kept = kept[: cfg.top]
    for i, r in enumerate(kept, 1):
        r["rank"] = r["n"] = i
    tm["total_s"] = round(time.perf_counter() - t0, 3)
    tm["audio_seconds"] = n
    return {"engine": f"highlight-v{ENGINE_VERSION}", "duration": duration, "config": asdict(cfg),
            "device": accel.summary(), "timings": tm, "notes": notes, "disclaimer": S.DISCLAIMER,
            "signals": {k: v for k, v in st.weights.items()}, "facecam_box": box,
            "shortlisted": len(cands), "rejected_low_evidence": len(weak), "highlights": kept,
            "duplicates": [{"start": r["start"], "end": r["end"], "duplicate_of": r["duplicate_of"],
                            "reason": r["duplicate_reason"]} for r in dropped]}


def _fmt(t):
    return f"{int(t // 60)}:{int(t % 60):02d}"


def write_outputs(d, res, source, info):
    d = Path(d)
    hl = res["highlights"]
    (d / "highlights.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    md = [f"# Highlights: {info.get('title', '')}", f"source: {info.get('webpage_url', '')}",
          f"engine: {res['engine']} · shortlisted {res['shortlisted']} · kept {len(hl)} · "
          f"duplicates removed {len(res['duplicates'])} · {res['timings']['total_s']}s", "",
          f"_{res['disclaimer']}_", ""]
    md += [f"- note: {x}" for x in res["notes"]] + ([""] if res["notes"] else [])
    for r in hl:
        s = r["scores"]
        md += [f"## {r['rank']}. {_fmt(r['start'])}-{_fmt(r['end'])} ({r['length']:.0f}s) "
               f"entertainment {s['entertainment']:.2f} · confidence {r['confidence']:.2f}",
               "scores: " + ", ".join(f"{k.replace('_', ' ')} {s[k]:.2f}" for k in S.DIMENSIONS[1:]),
               "why: " + "; ".join(r["evidence"]),
               f"boundaries: {r['boundary']['start_reason']} → {r['boundary']['end_reason']}",
               f"> {r['text'] or '(no speech)'}", ""]
    (d / "highlights.md").write_text("\n".join(md), encoding="utf-8")
    cands = [{"n": r["n"], "start": r["start"], "end": r["end"], "peak": r["peak"],
              "score": r["scores"]["entertainment"], "why": r["evidence"], "text": r["text"][:400],
              "confidence": r["confidence"]} for r in hl]
    (d / "candidates.json").write_text(json.dumps({"source": str(source), "info": info, "duration": res["duration"],
                                                   "candidates": cands, "engine": res["engine"],
                                                   "signals": res["signals"]}, ensure_ascii=False, indent=1), encoding="utf-8")
    cm = [f"# Candidates: {info.get('title', '')}", f"source: {info.get('webpage_url', '')}",
          f"ranked best first by {res['engine']} (see highlights.md for scores)", ""]
    for c in cands:
        cm += [f"## {c['n']}. {_fmt(c['start'])}-{_fmt(c['end'])} ({c['end'] - c['start']:.0f}s) score {c['score']:.2f}",
               f"why: {', '.join(c['why'])}", f"> {c['text']}", ""]
    (d / "candidates.md").write_text("\n".join(cm), encoding="utf-8")
