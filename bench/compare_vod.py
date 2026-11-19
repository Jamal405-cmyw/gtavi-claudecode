"""Blind comparison of detectors on one real VOD (no labels needed).

Step 1 - run:   python bench/compare_vod.py run <url-or-file> [--chat chat.json] [--top 5]
                       [--systems baseline,new] [--name my_vod] [--whisper-model auto] [--judge ollama]
  - downloads once, transcribes once (GPU if available) and gives every system identical inputs
  - runs each detector as its own process, recording wall time, CPU time, peak RAM, GPU util/VRAM
  - pools every system's top picks, renders a 480p preview of each, shuffles them, hides which
    system picked what, and writes bench/vod/<name>/rate.html

Step 2 - rate:  open rate.html, mark each clip Post / Maybe / No (+ optional 1-5 stars), click
                "Download ratings" -> ratings.json (keep it in the same folder)

Step 3 - score: python bench/compare_vod.py score bench/vod/<name> [--ratings ratings.json] [--labels]
  - un-blinds and writes report.md: per system precision@K (Post=1, Maybe=0.5), mean stars,
    clips only that system found, and the resource numbers from step 1
  - --labels also writes labels.json from the Post clips so the VOD joins the regression benchmark
    (pooled labels: they can only contain moments some system found, so recall is optimistic)
"""
import argparse, html, json, os, random, re, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "scripts")]
import run_benchmark as RB  # noqa: E402  (Monitor, SYSTEMS, job_dir_for, predictions)


def _same(a, b, tol=2.0):
    return abs(a["start"] - b["start"]) <= tol and abs(a["end"] - b["end"]) <= tol


def pool(picks):
    """picks: {system: [ {start,end}, ... ranked ]} -> list of items {start,end,by:{system: rank}}.
    Clips are merged only if both edges are within 2 s, so boundary differences stay visible to the rater."""
    items = []
    for s, ps in picks.items():
        for rank, p in enumerate(ps, 1):
            hit = next((it for it in items if _same(it, p)), None)
            if hit:
                hit["by"][s] = rank
            else:
                items.append({"start": round(p["start"], 2), "end": round(p["end"], 2), "by": {s: rank}})
    return items


def render_preview(video, start, end, out):
    r = subprocess.run([shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-to", f"{end:.2f}",
                        "-i", str(video), "-vf", "scale=-2:480", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out)], capture_output=True, text=True)
    return r.returncode == 0 and out.exists()


PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Blind clip rating</title>
<style>body{{font-family:system-ui,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;background:#111;color:#eee}}
.c{{display:flex;gap:16px;align-items:flex-start;border-bottom:1px solid #333;padding:16px 0}}video{{width:480px;max-width:60vw;background:#000}}
button{{margin:2px;padding:8px 14px;border:1px solid #555;background:#222;color:#eee;border-radius:6px;cursor:pointer}}
button.on{{background:#e8457f;border-color:#e8457f}}textarea{{width:100%;height:48px;background:#1a1a1a;color:#eee;border:1px solid #444}}
#bar{{position:sticky;top:0;background:#111;padding:8px 0;border-bottom:1px solid #333}}</style></head><body>
<div id="bar"><b>{title}</b> · {n} clips, shuffled, systems hidden · rated <span id="cnt">0</span>/{n}
 <button onclick="dl()">Download ratings</button></div>
<p>For each clip: would you post it as a short (with a hook and captions added)? Rate the moment <i>and</i> where it starts and ends.</p>
{rows}
<script>
const KEY="ratings_{key}";let R={{}};try{{R=JSON.parse(localStorage.getItem(KEY)||"{{}}")}}catch(e){{}}
function save(){{try{{localStorage.setItem(KEY,JSON.stringify(R))}}catch(e){{}};paint()}}
function set(id,k,v){{R[id]=R[id]||{{}};R[id][k]=v;save()}}
function paint(){{document.querySelectorAll("[data-id]").forEach(b=>{{const r=R[b.dataset.id]||{{}};
 b.classList.toggle("on",String(r[b.dataset.k])===b.dataset.v)}});
 document.querySelectorAll("textarea").forEach(t=>{{t.value=(R[t.dataset.n]||{{}}).note||""}});
 document.getElementById("cnt").textContent=Object.values(R).filter(r=>r.post).length}}
function dl(){{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([JSON.stringify(R,null,1)],{{type:"application/json"}}));
 a.download="ratings.json";a.click()}}
paint();
</script></body></html>"""

ROW = """<div class="c"><video controls preload="metadata" src="{src}"></video><div><b>{cid}</b> ({dur:.0f}s)<br>
{post}<br>{stars}<br><textarea data-n="{cid}" placeholder="note (optional)" oninput="set('{cid}','note',this.value)"></textarea></div></div>"""


def _buttons(cid, k, vals):
    js = lambda v: str(v) if isinstance(v, int) else f"'{v}'"  # single quotes: the attribute itself is double-quoted
    return "".join(f'<button data-id="{cid}" data-k="{k}" data-v="{v}" onclick="set(\'{cid}\',\'{k}\',{js(v)})">{lab}</button>'
                   for v, lab in vals)


def write_page(d, items, title):
    rows = "".join(ROW.format(src=html.escape(f"previews/{it['id']}.mp4"), cid=it["id"], dur=it["end"] - it["start"],
                              post=_buttons(it["id"], "post", [("yes", "Post"), ("maybe", "Maybe"), ("no", "No")]),
                              stars=_buttons(it["id"], "stars", [(i, "★" * i) for i in range(1, 6)])) for it in items)
    (d / "rate.html").write_text(PAGE.format(title=html.escape(title), n=len(items), rows=rows, key=d.name), encoding="utf-8")


def cmd_run(a):
    import common, moments
    from highlight import accel
    stem = a.src.split("v=")[-1].split("/")[-1] if common.is_url(a.src) else Path(a.src).stem
    name = a.name or re.sub(r"[^a-z0-9]+", "-", stem.lower())[:40].strip("-") or "vod"
    d = Path(a.out) / name
    d.mkdir(parents=True, exist_ok=True)
    timings, prep = {}, d / "prep"
    common.JOBS = prep
    jd = common.job_dir(a.src)
    t = time.perf_counter()
    video, info = moments.fetch(a.src, jd, a.max_minutes)
    timings["download_s"] = round(time.perf_counter() - t, 1)
    if not (d / "transcript.json").exists():
        model, device, _ = accel.whisper_choice(a.whisper_model)
        t = time.perf_counter()
        wav, _ = moments.audio_rms(video, jd)
        words = moments.transcribe(wav, jd, model=model, device=device)
        timings.update(transcribe_s=round(time.perf_counter() - t, 1), whisper=f"{model} on {device}")
        (d / "transcript.json").write_text(json.dumps(words), encoding="utf-8")
    chat = a.chat or next((str(p) for p in sorted(jd.glob("chat*.live_chat.json"))), None)
    if chat:
        shutil.copy(chat, d / ("chat.live_chat.json" if chat.endswith(".live_chat.json") else "chat.json"))
        chat = str(next(d.glob("chat*.json")).resolve())
    lab = {"video": str(Path(video).resolve()), "title": (info or {}).get("title") or Path(video).stem,
           "source_url": (info or {}).get("webpage_url") or a.src, "highlights": []}
    (d / "case.json").write_text(json.dumps(lab, indent=1), encoding="utf-8")
    runs, picks = {}, {}
    for s in a.systems.split(","):
        home = d / "runs" / s
        jdir = RB.job_dir_for(home, lab["video"])
        shutil.copy(d / "transcript.json", jdir / "transcript.json")
        cmd = RB.SYSTEMS[s](lab["video"], chat, a.cam, a.top)
        if s.startswith("new") and a.judge != "none" and "--judge" not in cmd:
            cmd += ["--judge", a.judge, "--judge-model", a.judge_model]
        t = time.perf_counter()
        p = subprocess.Popen(cmd, env=dict(os.environ, CLIP_DESK_HOME=str(home)), cwd=home,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        mon = RB.Monitor(p)
        so, se = p.communicate()
        res = mon.finish()
        res["wall_s"] = round(time.perf_counter() - t, 2)
        if p.returncode != 0:
            print(f"{s} failed:\n{(se or so)[-800:]}")
            runs[s] = dict(res, error=(se or so)[-800:])
            continue
        picks[s] = [{"start": float(x["start"]), "end": float(x["end"])} for x in RB.predictions(s, jdir)][: a.top]
        runs[s] = res
        print(f"{s}: {len(picks[s])} picks in {res['wall_s']}s (cpu {res['cpu_s']}s, peak RAM {res['peak_rss_mb']} MB, "
              f"GPU max {res['gpu_util_max']}%)")
    items = pool(picks)
    random.Random(a.seed).shuffle(items)
    (d / "previews").mkdir(exist_ok=True)
    for i, it in enumerate(items, 1):
        it["id"] = f"clip_{i:02d}"
        render_preview(lab["video"], it["start"], it["end"], d / "previews" / f"{it['id']}.mp4")
    # pool.json holds the un-blinding key: do not open it before rating
    (d / "pool.json").write_text(json.dumps({"items": items, "systems": list(picks), "top": a.top, "runs": runs,
                                             "prep": timings, "device": accel.summary()}, indent=1), encoding="utf-8")
    write_page(d, items, lab["title"])
    print(f"\n{len(items)} clips to rate -> {d / 'rate.html'}\n(then: python bench/compare_vod.py score {d})")
    return d


def score(items, ratings, systems, top):
    val = {"yes": 1.0, "maybe": 0.5, "no": 0.0}
    out = {}
    for s in systems:
        mine = sorted((it for it in items if s in it["by"]), key=lambda it: it["by"][s])[:top]
        rated = [ratings[it["id"]] for it in mine if ratings.get(it["id"], {}).get("post") in val]
        stars = [r["stars"] for r in rated if isinstance(r.get("stars"), (int, float))]
        only = [it for it in mine if len(it["by"]) == 1 and ratings.get(it["id"], {}).get("post") == "yes"]
        out[s] = {"picks": len(mine), "rated": len(rated),
                  "precision@K": round(sum(val[r["post"]] for r in rated) / top, 3) if top else None,
                  "post_rate": round(sum(r["post"] == "yes" for r in rated) / len(rated), 3) if rated else None,
                  "mean_stars": round(sum(stars) / len(stars), 2) if stars else None,
                  "post_only_this_system": len(only)}
    return out


def cmd_score(a):
    d = Path(a.dir)
    pool_ = json.loads((d / "pool.json").read_text(encoding="utf-8"))
    rp = Path(a.ratings) if a.ratings else d / "ratings.json"
    ratings = json.loads(rp.read_text(encoding="utf-8"))
    items, systems, top = pool_["items"], pool_["systems"], pool_["top"]
    sc = score(items, ratings, systems, top)
    case = json.loads((d / "case.json").read_text(encoding="utf-8"))
    L = [f"# Blind comparison: {case['title']}", "", f"source: {case['source_url']}",
         f"rated {sum(1 for it in items if ratings.get(it['id'], {}).get('post'))}/{len(items)} pooled clips · top {top} per system", "",
         "| system | precision@K | post rate | mean stars | Post clips only it found | wall s | CPU s | peak RAM MB | GPU max % | GPU mem max MB |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for s in systems:
        m, r = sc[s], pool_["runs"].get(s, {})
        f = lambda v: "–" if v is None else v
        L.append(f"| {s} | {f(m['precision@K'])} | {f(m['post_rate'])} | {f(m['mean_stars'])} | {m['post_only_this_system']} | "
                 f"{f(r.get('wall_s'))} | {f(r.get('cpu_s'))} | {f(r.get('peak_rss_mb'))} | {f(r.get('gpu_util_max'))} | {f(r.get('gpu_mem_max_mb'))} |")
    p = pool_.get("prep", {})
    L += ["", f"shared prep (not counted above): download {p.get('download_s', '–')} s, transcription "
              f"{p.get('transcribe_s', 'cached')} s ({p.get('whisper', '')}); device: {json.dumps(pool_.get('device', {}))}",
          "", "precision@K: Post = 1, Maybe = 0.5, No/unrated = 0, divided by K. One rater on one VOD is an anecdote, "
              "not a benchmark: compare several VODs before drawing conclusions."]
    (d / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    if a.labels:
        hl = [{"start": it["start"], "end": it["end"], "peak": round((it["start"] + it["end"]) / 2, 2), "kind": "rated_post"}
              for it in items if ratings.get(it["id"], {}).get("post") == "yes"]
        lab = {"video": case["video"], "highlights": hl, "source": "real", "mode": "pooled_blind_rating",
               "note": "labels pooled from system picks; moments no system found are missing (recall is optimistic)"}
        if (d / "chat.json").exists(): lab["chat"] = "chat.json"
        (d / "labels.json").write_text(json.dumps(lab, indent=1), encoding="utf-8")
    print((d / "report.md").read_text(encoding="utf-8"))
    return sc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("src"); r.add_argument("--chat"); r.add_argument("--top", type=int, default=5)
    r.add_argument("--systems", default="baseline,new"); r.add_argument("--name"); r.add_argument("--out", default=str(HERE / "vod"))
    r.add_argument("--cam", default="auto"); r.add_argument("--whisper-model", default="auto")
    r.add_argument("--max-minutes", type=int, default=180); r.add_argument("--seed", type=int, default=0)
    r.add_argument("--judge", default="none"); r.add_argument("--judge-model", default="qwen2.5vl:7b")
    s = sub.add_parser("score")
    s.add_argument("dir"); s.add_argument("--ratings"); s.add_argument("--labels", action="store_true")
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_score(a)


if __name__ == "__main__":
    main()
