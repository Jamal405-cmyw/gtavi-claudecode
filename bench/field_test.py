"""Controlled real-VOD field test: original vs improved vs improved+VLM, reviewed blind.

Configurations (identical source video, transcript and chat file; 5 clips each; same layout/captions):
  A  original:  frozen bench/baseline/moments.py + frozen bench/baseline/cut.py (original editing)
  B  improved:  scripts/highlights.py (speech-aware edges)  + scripts/cut.py --reframe auto
  C  improved+VLM: as B, plus --judge ollama (local VLM re-ranks the top clips)

Stages (each is a separate command so it can be approved on its own):
  python bench/field_test.py plan    <url> [--minutes 90]          read-only: VOD length, disk estimate
  python bench/field_test.py prepare <url> --name vod1 [--minutes 90] [--chat file] [--transcript file]
         downloads the VOD once, the chat once (TwitchDownloaderCLI), transcribes once (Whisper, GPU)
  python bench/field_test.py detect  vod1 [--configs A,B,C] [--judge-model qwen2.5vl:7b]
  python bench/field_test.py render  vod1 [--layout react] [--credit @streamer]
  python bench/field_test.py page    vod1                              blind review page
  python bench/field_test.py score   vod1                              after rating: report.md

Everything lives in bench/field/<name>/ (git-ignored). Every step's wall time, CPU time, peak RAM and
peak GPU utilisation/memory (nvidia-smi) and any failure is appended to log.json. The detector code is
hashed at `prepare`; `score` refuses to call results clean if that code changed in between (no tuning
on the VOD being measured).
"""
import argparse, gzip, hashlib, html, json, os, random, shutil, subprocess, sys, time, types
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "scripts")]
import run_benchmark as RB  # noqa: E402  (Monitor, job_dir_for)

CONFIGS = ("A", "B", "C")
LABEL = {"A": "original detector + original editing", "B": "improved detector + speech edges + reframing",
         "C": "improved + local VLM ranking"}
HASHED = ["bench/baseline/moments.py", "bench/baseline/cut.py", "bench/baseline/facecam.py", "bench/baseline/common.py",
          "scripts/highlights.py", "scripts/cut.py", "scripts/reframe.py", "scripts/facecam.py", "scripts/common.py",
          "scripts/moments.py"] + sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "scripts" / "highlight").glob("*.py"))


def code_hashes():
    # line endings normalised so a CRLF checkout (Git for Windows default) hashes like the repository
    return {f: hashlib.sha256((ROOT / f).read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]
            for f in HASHED if (ROOT / f).exists()}


def commit():
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or "?"


def fdir(name):
    return HERE / "field" / name


def jload(p, default=None):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def jsave(p, obj):
    Path(p).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


def log_step(d, rec):
    log = jload(d / "log.json", [])
    rec["at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log.append(rec)
    jsave(d / "log.json", log)
    status = rec.get("status")
    print(f"[{rec['step']}] {status} {rec.get('wall_s', '')}s"
          + (f" gpu_mem_max={rec.get('gpu_mem_max_mb')}MB" if rec.get("gpu_mem_max_mb") else "")
          + (f" ERROR {rec.get('error', '')[:200]}" if status != "ok" else ""))


def measure_inproc(d, step, fn):
    """Run fn() in this process, sampling this process tree (incl. yt-dlp/ffmpeg children) and the GPU."""
    t = time.perf_counter()
    mon = RB.Monitor(types.SimpleNamespace(pid=os.getpid()))
    try:
        out = fn()
        rec = dict(step=step, status="ok")
    except Exception as ex:  # recorded, not hidden
        out, rec = None, dict(step=step, status="failed", error=f"{type(ex).__name__}: {ex}")
    rec.update(mon.finish(), wall_s=round(time.perf_counter() - t, 1))
    log_step(d, rec)
    return out, rec


def measure_cmd(d, step, cmd, env=None, cwd=None):
    t = time.perf_counter()
    p = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    mon = RB.Monitor(p)
    so, se = p.communicate()
    rec = dict(step=step, status="ok" if p.returncode == 0 else "failed")
    if p.returncode != 0:
        rec["error"] = (se or so).strip()[-600:]
    rec.update(mon.finish(), wall_s=round(time.perf_counter() - t, 1))
    log_step(d, rec)
    return rec, so


# ------------------------------------------------------------------ plan (read-only)
def cmd_plan(a):
    import moments
    from common import ytdlp_json
    info = ytdlp_json(a.src)  # metadata only, no media download
    dur = float(info.get("duration") or 0)
    used = min(dur, a.minutes * 60) if dur else a.minutes * 60
    fmts = [f for f in info.get("formats", []) if (f.get("height") or 0) <= 1080 and f.get("tbr")]
    tbr = max((f["tbr"] for f in fmts), default=6000.0)  # kbit/s of the chosen <=1080p rendition
    gb_video = tbr * 1000 / 8 * used / 1e9
    gb_wav = used * 16000 * 2 * 4 / 1e9  # audio.wav in prep + 3 config job folders
    print(json.dumps({"title": info.get("title"), "channel": info.get("uploader") or info.get("channel"),
                      "vod_seconds": dur, "analysed_seconds": used, "bitrate_kbps_1080p": round(tbr),
                      "est_disk_gb": {"video": round(gb_video, 2), "audio_wavs": round(gb_wav, 2),
                                      "renders_and_blind_copies": 0.9, "chat": 0.1},
                      "note": "whisper model (1.62 GB) and the optional VLM (about 6 GB) are separate one-time downloads"},
                     indent=1))
    _ = moments  # imported to fail early if the environment is incomplete


# ------------------------------------------------------------------ prepare
def cmd_prepare(a):
    import common, moments
    from highlight import accel
    d = fdir(a.name)
    d.mkdir(parents=True, exist_ok=True)
    case = jload(d / "case.json", {})
    if case and case.get("src") != a.src:
        sys.exit(f"{d} already belongs to {case.get('src')}; pick another --name")
    case.update(src=a.src, minutes=a.minutes, commit=commit(), code=code_hashes(), device=accel.summary(),
                created=case.get("created") or datetime.now(timezone.utc).isoformat(timespec="seconds"))
    jsave(d / "case.json", case)
    common.JOBS = d / "prep"
    jd = common.job_dir(a.src)
    got, rec = measure_inproc(d, "prepare.download", lambda: moments.fetch(a.src, jd, a.minutes))
    if not got:
        sys.exit("download failed; see log.json")
    video, info = got
    case.update(video=str(Path(video).resolve()), title=(info or {}).get("title"),
                channel=(info or {}).get("channel") or (info or {}).get("uploader"))
    # ---- chat (one plain JSON file shared by all configs: the frozen baseline cannot read .gz)
    chat_out = d / "chat.json"
    if a.chat:
        src = Path(a.chat)
        data = src.read_bytes()
        chat_out.write_bytes(gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data)
        log_step(d, dict(step="prepare.chat", status="ok", source=str(src)))
    elif not chat_out.exists():
        import twitch

        def fetch_chat():
            gz = twitch.vod_chat(a.src, d / "chat_dl.json", end_seconds=a.minutes * 60)
            chat_out.write_bytes(gzip.decompress(gz.read_bytes()))
            return gz
        measure_inproc(d, "prepare.chat", fetch_chat)
    case["chat"] = str(chat_out) if chat_out.exists() else None
    # ---- transcript once (shared)
    if a.transcript:
        shutil.copy(a.transcript, d / "transcript.json")
        log_step(d, dict(step="prepare.transcribe", status="ok", source=a.transcript))
    elif not (d / "transcript.json").exists():
        model, device, _ = accel.whisper_choice(a.whisper_model)

        def tx():
            wav, _ = moments.audio_rms(video, jd)
            words = moments.transcribe(wav, jd, model=model, device=device)
            jsave(d / "transcript.json", words)
            return len(words)
        n, rec = measure_inproc(d, f"prepare.transcribe[{model} on {device}]", tx)
        case["whisper"] = f"{model} on {device}"
    jsave(d / "case.json", case)
    print(f"prepared {d}: video {case['video']}, chat {'yes' if case['chat'] else 'NO'}, transcript yes")


# ------------------------------------------------------------------ detect
def _job(d, cfg, video):
    home = d / "runs" / cfg
    home.mkdir(parents=True, exist_ok=True)
    jd = RB.job_dir_for(home, video)
    shutil.copy(d / "transcript.json", jd / "transcript.json")
    return home, jd


def cmd_detect(a):
    d = fdir(a.name)
    case = jload(d / "case.json")
    if not case or not (d / "transcript.json").exists():
        sys.exit("run prepare first")
    video, chat = case["video"], case.get("chat")
    picks = jload(d / "picks.json", {})
    for cfg in a.configs.split(","):
        home, jd = _job(d, cfg, video)
        chat_args = ["--chat", chat] if chat else []
        if cfg == "A":
            cmd = [sys.executable, str(HERE / "baseline" / "moments.py"), video, "--top", str(a.clips)] + chat_args
        else:
            cmd = [sys.executable, str(ROOT / "scripts" / "highlights.py"), video, "--top", str(a.clips), "--no-transcribe",
                   "--cam", a.cam, "--no-twitch-chat"] + chat_args
            if cfg == "C":
                cmd += ["--judge", "ollama", "--judge-model", a.judge_model, "--judge-host", a.judge_host,
                        "--judge-top", str(max(6, a.clips + 1))]
        rec, _ = measure_cmd(d, f"detect.{cfg}", cmd, env=dict(os.environ, CLIP_DESK_HOME=str(home)), cwd=home)
        if rec["status"] != "ok":
            picks[cfg] = {"error": rec.get("error")}
            continue
        if cfg == "A":
            c = jload(jd / "candidates.json")["candidates"]
            ranked = sorted(c, key=lambda x: -x["score"])
            picks[cfg] = {"job": str(jd), "picks": [{"n": x["n"], "rank": i, "start": x["start"], "end": x["end"],
                                                     "score": x["score"]} for i, x in enumerate(ranked, 1)]}
        else:
            h = jload(jd / "highlights.json")
            notes = h.get("notes", [])
            picks[cfg] = {"job": str(jd), "notes": notes, "timings": h.get("timings"),
                          "picks": [{"n": x["n"], "rank": x["rank"], "start": x["start"], "end": x["end"],
                                     "score": x["scores"]["entertainment"], "judge": x.get("judge")} for x in h["highlights"]]}
            if cfg == "C" and not any(p.get("judge") and "error" not in p["judge"] for p in picks[cfg]["picks"]):
                picks[cfg]["error"] = "VLM judge did not run: " + "; ".join(n for n in notes if "judge" in n.lower())
                log_step(d, dict(step="detect.C.judge", status="failed", error=picks[cfg]["error"]))
        jsave(d / "picks.json", picks)
    for cfg, p in picks.items():
        print(cfg, LABEL[cfg], "->", p.get("error") or [(round(x["start"]), round(x["end"])) for x in p["picks"]])


# ------------------------------------------------------------------ render
def cmd_render(a):
    d = fdir(a.name)
    picks = jload(d / "picks.json")
    if not picks:
        sys.exit("run detect first")
    out = d / "renders"; out.mkdir(exist_ok=True)
    renders = jload(d / "renders.json", {})
    done_by_span = {}  # B and C use identical editing code: the same span is rendered once
    for cfg in CONFIGS:
        p = picks.get(cfg)
        if not p or p.get("error"):
            continue
        jd = Path(p["job"])
        for x in p["picks"]:
            key = f"{cfg}_{x['rank']}"
            span = (round(x["start"], 2), round(x["end"], 2))
            if cfg in ("B", "C") and ("BC",) + span in done_by_span:
                renders[key] = dict(done_by_span[("BC",) + span], shared_with=done_by_span[("BC",) + span]["key"])
                continue
            script = HERE / "baseline" / "cut.py" if cfg == "A" else ROOT / "scripts" / "cut.py"
            cmd = [sys.executable, str(script), str(jd), "--pick", str(x["n"]), "--layout", a.layout, "--cam", a.cam]
            if a.credit:
                cmd += ["--credit", a.credit]
            if cfg != "A":
                cmd += ["--reframe", "auto"]
            rec, so = measure_cmd(d, f"render.{key}", cmd, cwd=jd)
            mp4 = jd / "edits" / f"clip_{x['n']}.mp4"
            if rec["status"] != "ok" or not mp4.exists():
                renders[key] = {"key": key, "error": rec.get("error", "no output")}
                continue
            dst = out / f"{key}.mp4"
            shutil.copy(mp4, dst)
            meta = jload(jd / "edits" / f"clip_{x['n']}.json", {})
            renders[key] = {"key": key, "file": str(dst), "source_start": x["start"], "source_end": x["end"],
                            "layout_used": meta.get("layout"), "cam": meta.get("cam"), "reframe": meta.get("reframe"),
                            "facecam_log": [ln for ln in so.splitlines() if ln.startswith(("facecam:", "reframe:"))]}
            if cfg in ("B", "C"):
                done_by_span[("BC",) + span] = renders[key]
            jsave(d / "renders.json", renders)
    jsave(d / "renders.json", renders)
    ok = sum(1 for r in renders.values() if r.get("file"))
    print(f"{ok} renders ok, {sum(1 for r in renders.values() if r.get('error'))} failed -> {out}")


# ------------------------------------------------------------------ blind page
QUESTIONS = [
    ("sel", "entertaining", "Is the moment genuinely entertaining?", [(1, "1"), (2, "2"), (3, "3"), (4, "4"), (5, "5")]),
    ("sel", "setup", "Is the setup included?", [("yes", "Yes"), ("partly", "Partly"), ("no", "No")]),
    ("sel", "payoff", "Is the payoff included?", [("yes", "Yes"), ("partly", "Partly"), ("no", "No")]),
    ("sel", "post", "Would you post this moment?", [("yes", "Post"), ("maybe", "Maybe"), ("no", "No")]),
    ("edit", "start_ok", "Does it start at a natural point?", [("yes", "Yes"), ("no", "No")]),
    ("edit", "end_ok", "Does it end at a natural point?", [("yes", "Yes"), ("no", "No")]),
    ("edit", "facecam", "Facecam visible and well framed?", [("good", "Good"), ("poor", "Poor"), ("missing", "Missing")]),
    ("edit", "gameplay", "Gameplay/player visible in the gameplay panel?", [("good", "Good"), ("poor", "Poor"), ("missing", "Missing")]),
    ("edit", "subs", "Subtitles accurate?", [("accurate", "Accurate"), ("minor", "Minor errors"), ("major", "Major errors"), ("none", "No subtitles")]),
]

PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Blind clip review</title><style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:20px auto;padding:0 16px;background:#111;color:#eee}}
.c{{display:flex;gap:20px;border-bottom:1px solid #333;padding:18px 0}}video{{height:560px;background:#000}}
.q{{margin:6px 0}}.q span{{display:inline-block;width:330px}}h4{{margin:10px 0 4px;color:#aaa;font-weight:600}}
button{{margin:1px;padding:5px 10px;border:1px solid #555;background:#222;color:#eee;border-radius:5px;cursor:pointer}}
button.on{{background:#e8457f;border-color:#e8457f}}textarea{{width:100%;height:44px;background:#1a1a1a;color:#eee;border:1px solid #444}}
#bar{{position:sticky;top:0;background:#111;padding:8px 0;border-bottom:1px solid #333;z-index:2}}</style></head><body>
<div id="bar"><b>Blind review</b> · {n} clips in random order · configurations hidden · complete <span id="cnt">0</span>/{n}
 <button onclick="dl()">Download ratings</button></div>
<p>Rate each clip on its own. <b>Selection</b> = is the moment worth posting. <b>Editing</b> = how it is cut and framed.
Ratings are kept in this browser until you download them as <code>ratings.json</code> into this folder.</p>
{rows}
<script>
const KEY="field_{key}";let R={{}};try{{R=JSON.parse(localStorage.getItem(KEY)||"{{}}")}}catch(e){{}}
const NQ={nq};
function save(){{try{{localStorage.setItem(KEY,JSON.stringify(R))}}catch(e){{}};paint()}}
function set(id,k,v){{R[id]=R[id]||{{}};R[id][k]=v;save()}}
function paint(){{document.querySelectorAll("[data-id]").forEach(b=>{{const r=R[b.dataset.id]||{{}};
 b.classList.toggle("on",String(r[b.dataset.k])===b.dataset.v)}});
 document.querySelectorAll("textarea").forEach(t=>{{t.value=(R[t.dataset.n]||{{}}).note||""}});
 document.getElementById("cnt").textContent=Object.values(R).filter(r=>Object.keys(r).filter(k=>k!=="note").length>=NQ).length}}
function dl(){{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(R,null,1)],{{type:"application/json"}}));
 a.download="ratings.json";a.click()}}
paint();
</script></body></html>"""


def _btns(cid, k, opts):
    js = lambda v: str(v) if isinstance(v, int) else f"'{v}'"
    return "".join(f'<button data-id="{cid}" data-k="{k}" data-v="{v}" onclick="set(\'{cid}\',\'{k}\',{js(v)})">{html.escape(lab)}</button>'
                   for v, lab in opts)


def cmd_page(a):
    d = fdir(a.name)
    renders = jload(d / "renders.json")
    if not renders:
        sys.exit("run render first")
    files = {}
    for key, r in renders.items():
        if r.get("file"):
            files.setdefault(r["file"], []).append(key)  # a B/C span rendered once is one review item
    items = [{"file": f, "keys": sorted(ks)} for f, ks in sorted(files.items())]
    random.Random(a.seed).shuffle(items)
    blind = d / "blind"
    if blind.exists():
        shutil.rmtree(blind)
    blind.mkdir()
    rows = []
    for i, it in enumerate(items, 1):
        cid = f"clip_{i:02d}"
        it["id"] = cid
        shutil.copy(it["file"], blind / f"{cid}.mp4")
        qs = []
        for sec, k, label, opts in QUESTIONS:
            if sec == "sel" and k == "entertaining":
                qs.append("<h4>Selection</h4>")
            if sec == "edit" and k == "start_ok":
                qs.append("<h4>Editing</h4>")
            qs.append(f'<div class="q"><span>{html.escape(label)}</span>{_btns(cid, k, opts)}</div>')
        qs.append(f'<textarea data-n="{cid}" placeholder="notes (optional)" oninput="set(\'{cid}\',\'note\',this.value)"></textarea>')
        rows.append(f'<div class="c"><video controls preload="metadata" src="{cid}.mp4"></video><div><b>{cid}</b>{"".join(qs)}</div></div>')
    (blind / "rate.html").write_text(PAGE.format(n=len(items), rows="".join(rows), key=a.name, nq=len(QUESTIONS)), encoding="utf-8")
    # un-blinding key lives OUTSIDE the blind folder
    jsave(d / "key.json", {"seed": a.seed, "items": [{"id": it["id"], "renders": it["keys"]} for it in items]})
    print(f"{len(items)} clips -> {blind / 'rate.html'}  (save ratings.json into {blind})")


# ------------------------------------------------------------------ score
SEL_VAL = {"yes": 1.0, "partly": 0.5, "maybe": 0.5, "no": 0.0}
EDIT_VAL = {"yes": 1.0, "no": 0.0, "good": 1.0, "poor": 0.5, "missing": 0.0, "accurate": 1.0, "minor": 0.5, "major": 0.0}


def _mean(v):
    v = [x for x in v if x is not None]
    return round(sum(v) / len(v), 2) if v else None


def spearman(xs, ys):
    """Rank correlation (average ranks for ties); None if fewer than 3 pairs or no variance."""
    if len(xs) < 3:
        return None

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    if sx == 0 or sy == 0:
        return None
    return round(sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (sx * sy), 2)


def summarise(picks, renders, key, ratings, clips):
    by_render = {k: it["id"] for it in key["items"] for k in it["renders"]}
    out = {}
    for cfg in CONFIGS:
        p = picks.get(cfg)
        if not p or p.get("error"):
            out[cfg] = {"error": (p or {}).get("error", "not run")}
            continue
        rows = []
        for x in p["picks"][:clips]:
            rk = f"{cfg}_{x['rank']}"
            cid = by_render.get(rk)
            r = ratings.get(cid, {}) if cid else {}
            rows.append({"rank": x["rank"], "clip": cid, "render_ok": bool(renders.get(rk, {}).get("file")), **r})
        ent = [r.get("entertaining") for r in rows]
        sel = {"rated": sum(1 for r in rows if r.get("post")),
               "mean_entertaining_1to5": _mean(ent),
               "precision_at_k": round(sum(SEL_VAL.get(r.get("post"), 0) for r in rows) / clips, 2),
               "setup_included": _mean([SEL_VAL.get(r.get("setup")) for r in rows]),
               "payoff_included": _mean([SEL_VAL.get(r.get("payoff")) for r in rows]),
               "rank_vs_entertaining_spearman": spearman([-r["rank"] for r in rows if r.get("entertaining")],
                                                         [r["entertaining"] for r in rows if r.get("entertaining")])}
        ed = {k: _mean([EDIT_VAL.get(r.get(k)) for r in rows]) for k in ("start_ok", "end_ok", "facecam", "gameplay", "subs")}
        ed["render_failures"] = sum(1 for r in rows if not r["render_ok"])
        out[cfg] = {"selection": sel, "editing": ed, "rows": rows}
    # VLM effect: what C changed relative to B (same detector, different ranking)
    if "rows" in out.get("B", {}) and "rows" in out.get("C", {}):
        spans = lambda cfg: {(round(x["start"], 1), round(x["end"], 1)) for x in picks[cfg]["picks"][:clips]}
        only_c, only_b = spans("C") - spans("B"), spans("B") - spans("C")

        def ent_of(cfg, sp):
            return [r.get("entertaining") for r, x in zip(out[cfg]["rows"], picks[cfg]["picks"])
                    if (round(x["start"], 1), round(x["end"], 1)) in sp]
        out["vlm_effect"] = {"same_clips": len(spans("B") & spans("C")), "swapped_in_by_vlm": len(only_c),
                             "mean_entertaining_swapped_in": _mean(ent_of("C", only_c)),
                             "mean_entertaining_swapped_out": _mean(ent_of("B", only_b)),
                             "precision_at_k_B": out["B"]["selection"]["precision_at_k"],
                             "precision_at_k_C": out["C"]["selection"]["precision_at_k"]}
    return out


def cmd_score(a):
    d = fdir(a.name)
    case, picks, renders, key = (jload(d / f) for f in ("case.json", "picks.json", "renders.json", "key.json"))
    ratings = jload(Path(a.ratings) if a.ratings else d / "blind" / "ratings.json")
    if not all((case, picks, renders, key)) or ratings is None:
        sys.exit("need case/picks/renders/key from earlier steps and blind/ratings.json")
    res = summarise(picks, renders, key, ratings, a.clips)
    changed = [f for f, h in code_hashes().items() if case["code"].get(f) not in (None, h)]
    log = jload(d / "log.json", [])
    L = [f"# Field test: {case.get('title')}", "", f"source: {case['src']} · analysed first {case['minutes']} min · "
         f"commit {case['commit']} · whisper {case.get('whisper', 'cached')} · rated {len(ratings)} clips", ""]
    L += ["**Code changed since prepare: results are NOT a clean baseline measurement:** " + ", ".join(changed), ""] if changed else \
         ["Detector/editing code unchanged between prepare and scoring (no tuning on this VOD).", ""]
    f = lambda v: "–" if v is None else str(v)                      # plain numbers (seconds, MB, 1-5 means)
    pc = lambda v: "–" if v is None else f"{v:.0%}"                  # rates in 0..1
    L += ["## 1. Highlight selection", "", "| config | mean entertaining (1-5) | precision@K (Post=1, Maybe=0.5) | setup included | payoff included | rank vs rating (Spearman) |",
          "|---|---|---|---|---|---|"]
    for cfg in CONFIGS:
        r = res[cfg]
        if "error" in r:
            L.append(f"| {cfg} {LABEL[cfg]} | FAILED: {str(r['error'])[:120]} | | | | |"); continue
        s = r["selection"]
        L.append(f"| {cfg} {LABEL[cfg]} | {f(s['mean_entertaining_1to5'])} | {pc(s['precision_at_k'])} | {pc(s['setup_included'])} | "
                 f"{pc(s['payoff_included'])} | {f(s['rank_vs_entertaining_spearman'])} |")
    if "vlm_effect" in res:
        v = res["vlm_effect"]
        L += ["", f"VLM effect (C vs B, same detector): {v['same_clips']} clips identical, {v['swapped_in_by_vlm']} swapped in; "
                  f"mean entertaining of swapped-in {f(v['mean_entertaining_swapped_in'])} vs swapped-out {f(v['mean_entertaining_swapped_out'])}; "
                  f"precision@5 B {pc(v['precision_at_k_B'])} → C {pc(v['precision_at_k_C'])}."]
    L += ["", "## 2. Editing quality", "", "| config | natural start | natural end | facecam framed | gameplay/player visible | subtitles accurate | render failures |",
          "|---|---|---|---|---|---|---|"]
    for cfg in CONFIGS:
        r = res[cfg]
        if "error" in r:
            continue
        e = r["editing"]
        L.append(f"| {cfg} | {pc(e['start_ok'])} | {pc(e['end_ok'])} | {pc(e['facecam'])} | {pc(e['gameplay'])} | {pc(e['subs'])} | {e['render_failures']} |")
    L += ["", "## 3. Processing time, GPU and failures", "", "| step | status | wall s | CPU s | peak RAM MB | GPU util max % | GPU mem max MB |",
          "|---|---|---|---|---|---|---|"]
    for s in log:
        L.append(f"| {s['step']} | {s['status']} | {f(s.get('wall_s'))} | {f(s.get('cpu_s'))} | {f(s.get('peak_rss_mb'))} | "
                 f"{f(s.get('gpu_util_max'))} | {f(s.get('gpu_mem_max_mb'))} |")
    fails = [s for s in log if s["status"] != "ok"]
    L += ["", f"Failures: {len(fails)}"] + [f"- {s['step']}: {s.get('error', '')[:300]}" for s in fails]
    L += ["", "## Per clip", "", "| clip | renders | entertaining | post | setup | payoff | start | end | facecam | gameplay | subs | note |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for it in key["items"]:
        r = ratings.get(it["id"], {})
        spans = [f"{k} {renders[k].get('source_start', 0):.1f}-{renders[k].get('source_end', 0):.1f}s" for k in it["renders"]]
        L.append(f"| {it['id']} | {', '.join(spans)} | " + " | ".join(str(r.get(k, '–')) for k in
                 ("entertaining", "post", "setup", "payoff", "start_ok", "end_ok", "facecam", "gameplay", "subs"))
                 + f" | {html.escape(str(r.get('note', '')))[:80]} |")
    L += ["", "One VOD and one rater: an anecdote, not a benchmark. Clip length and framing style can hint at the configuration "
              "despite blinding. Source timestamps above are positions in the original VOD."]
    jsave(d / "results.json", res)
    (d / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print((d / "report.md").read_text(encoding="utf-8"))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("plan"); s.add_argument("src"); s.add_argument("--minutes", type=int, default=90)
    s = sub.add_parser("prepare"); s.add_argument("src"); s.add_argument("--name", required=True)
    s.add_argument("--minutes", type=int, default=90); s.add_argument("--chat"); s.add_argument("--transcript")
    s.add_argument("--whisper-model", default="auto")
    s = sub.add_parser("detect"); s.add_argument("name"); s.add_argument("--configs", default="A,B,C")
    s.add_argument("--clips", type=int, default=5); s.add_argument("--cam", default="auto")
    s.add_argument("--judge-model", default="qwen2.5vl:7b"); s.add_argument("--judge-host", default="http://127.0.0.1:11434")
    s = sub.add_parser("render"); s.add_argument("name"); s.add_argument("--layout", default="react")
    s.add_argument("--cam", default="auto"); s.add_argument("--credit", default="")
    s = sub.add_parser("page"); s.add_argument("name"); s.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("score"); s.add_argument("name"); s.add_argument("--ratings"); s.add_argument("--clips", type=int, default=5)
    a = ap.parse_args(argv)
    return {"plan": cmd_plan, "prepare": cmd_prepare, "detect": cmd_detect, "render": cmd_render,
            "page": cmd_page, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
