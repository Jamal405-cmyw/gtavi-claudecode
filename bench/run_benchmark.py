"""Benchmark the new highlight engine against the original detector (frozen in bench/baseline/).

usage: python bench/run_benchmark.py [--cases bench/cases] [--out bench/results] [--top 8]
                                     [--systems baseline,new,new_novisual] [--whisper-model auto]

A case is a folder with labels.json:
  {"video": "source.mp4", "highlights": [{"start": 61.0, "end": 79.5, "peak": 70.0}, ...],
   "cam": [x, y, w, h] (optional), "chat": "chat.json" (optional)}
plus optional transcript.json (word timestamps) and chat.json/chat.live_chat.json.
If transcript.json is missing it is created ONCE with faster-whisper and shared by every system,
so transcription cost and quality never differ between systems (its time is reported separately).

Every system runs as a separate process on identical inputs; wall time, CPU time, peak RSS and
(if nvidia-smi exists) GPU utilisation/memory are sampled while it runs.
"""
import argparse, json, os, shutil, subprocess, sys, threading, time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "scripts")]
import metrics  # noqa: E402

SYSTEMS = {
    "baseline": lambda v, chat, cam, top: [sys.executable, str(HERE / "baseline" / "moments.py"), v, "--top", str(top)]
    + (["--chat", chat] if chat else []),
    "new": lambda v, chat, cam, top: [sys.executable, str(ROOT / "scripts" / "highlights.py"), v, "--top", str(top),
                                      "--no-transcribe", "--cam", cam] + (["--chat", chat] if chat else []),
    "new_novisual": lambda v, chat, cam, top: [sys.executable, str(ROOT / "scripts" / "highlights.py"), v, "--top", str(top),
                                               "--no-transcribe", "--no-visual"] + (["--chat", chat] if chat else []),
    # opt-in: needs a local Ollama server with the model pulled (--systems baseline,new,new_judge)
    "new_judge": lambda v, chat, cam, top: [sys.executable, str(ROOT / "scripts" / "highlights.py"), v, "--top", str(top),
                                            "--no-transcribe", "--cam", cam, "--judge", "ollama",
                                            "--judge-model", os.environ.get("JUDGE_MODEL", "qwen2.5vl:7b")]
    + (["--chat", chat] if chat else []),
}


class Monitor:
    """Samples process-tree RSS/CPU (psutil) and GPU (nvidia-smi) while a command runs."""

    def __init__(self, proc):
        self.proc, self.stop = proc, threading.Event()
        self.rss, self.cpu, self.gpu_util, self.gpu_mem = 0, 0.0, [], []
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start()
        self.gt = threading.Thread(target=self._gpu, daemon=True)
        if shutil.which("nvidia-smi"):
            self.gt.start()

    def _run(self):
        try:
            import psutil
        except ImportError:
            return
        try:
            root = psutil.Process(self.proc.pid)
        except psutil.Error:
            return
        cpu = {}
        while not self.stop.is_set():
            try:
                procs = [root] + root.children(recursive=True)
                rss = 0
                for p in procs:
                    try:
                        rss += p.memory_info().rss
                        t = p.cpu_times(); cpu[p.pid] = t.user + t.system
                    except psutil.Error:
                        pass
                self.rss = max(self.rss, rss)
                self.cpu = sum(cpu.values())
            except psutil.Error:
                break
            time.sleep(0.05)

    def _gpu(self):
        while not self.stop.is_set():
            r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True)
            try:
                u, m = r.stdout.strip().split("\n")[0].split(",")
                self.gpu_util.append(float(u)); self.gpu_mem.append(float(m))
            except ValueError:
                pass
            time.sleep(0.5)

    def finish(self):
        self.stop.set(); self.t.join(2)
        return {"peak_rss_mb": round(self.rss / 2 ** 20, 1) if self.rss else None,
                "cpu_s": round(self.cpu, 2) if self.cpu else None,
                "gpu_util_max": max(self.gpu_util) if self.gpu_util else None,
                "gpu_mem_max_mb": max(self.gpu_mem) if self.gpu_mem else None}


def job_dir_for(home, src):
    import common
    common.JOBS = Path(home) / "jobs"
    return common.job_dir(src)


def ensure_transcript(case, video, whisper_model):
    tp = case / "transcript.json"
    if tp.exists():
        return 0.0
    import moments
    from highlight import accel
    tmp = case / "_tx"; tmp.mkdir(exist_ok=True)
    t = time.perf_counter()
    wav, _ = moments.audio_rms(video, tmp)
    model, device, _ = accel.whisper_choice(whisper_model)
    moments.transcribe(wav, tmp, model=model, device=device)
    shutil.copy(tmp / "transcript.json", tp); shutil.rmtree(tmp)
    return round(time.perf_counter() - t, 1)


def predictions(system, d):
    if system == "baseline":
        c = json.loads((d / "candidates.json").read_text(encoding="utf-8"))["candidates"]
        return sorted(c, key=lambda x: -x["score"])  # the original's own ranking (it lists by start time)
    return json.loads((d / "highlights.json").read_text(encoding="utf-8"))["highlights"]


def run_case(case, systems, out, top, whisper_model):
    lab = json.loads((case / "labels.json").read_text(encoding="utf-8"))
    video = str((case / lab.get("video", "source.mp4")).resolve())
    chat = lab.get("chat") or next((p.name for p in (case / "chat.json", case / "chat.live_chat.json") if p.exists()), None)
    chat = str((case / chat).resolve()) if chat else None
    cam = ":".join(f"{v:.4f}" for v in lab["cam"]) if lab.get("cam") else "auto"
    tx_s = ensure_transcript(case, video, whisper_model)
    dur = float(lab.get("duration") or 0)
    rows = {}
    for s in systems:
        home = out / s / case.name
        d = job_dir_for(home, video)
        shutil.copy(case / "transcript.json", d / "transcript.json")
        env = dict(os.environ, CLIP_DESK_HOME=str(home))
        t = time.perf_counter()
        p = subprocess.Popen(SYSTEMS[s](video, chat, cam, top), env=env, cwd=home, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
        mon = Monitor(p)
        so, se = p.communicate()
        wall = time.perf_counter() - t
        res = mon.finish()
        if p.returncode != 0:
            rows[s] = {"error": (se or so)[-600:], "wall_s": round(wall, 2)}
            continue
        if not dur:
            dur = json.loads((d / "candidates.json").read_text())["duration"]
        preds = [{"start": float(x["start"]), "end": float(x["end"])} for x in predictions(s, d)]
        m = metrics.evaluate(preds, lab["highlights"])
        m.update(res, wall_s=round(wall, 2), sec_per_video_min=round(wall / (dur / 60), 3) if dur else None)
        m["preds"] = preds
        rows[s] = m
    return {"case": case.name, "mode": lab.get("mode", "real"), "source": lab.get("source", "real"),
            "duration_s": dur, "transcribe_s": tx_s, "systems": rows}


def fmt(v, pct=False):
    if v is None:
        return "–"
    return f"{v * 100:.0f}%" if pct else (f"{v:.2f}" if isinstance(v, float) else str(v))


def report(results, systems, meta):
    cols = [("p@R", True), ("r@R", True), ("p@5", True), ("ap", False), ("ndcg@5", False), ("setup_included", True),
            ("payoff_included", True), ("start_err_s", False), ("end_err_s", False), ("mean_len_s", False),
            ("sec_per_video_min", False), ("cpu_s", False), ("peak_rss_mb", False), ("gpu_util_max", False)]
    lines = [f"# Highlight benchmark — {meta['when']}", "",
             f"cases: {len(results)} ({', '.join(sorted({r['source'] for r in results}))}) · top {meta['top']} · "
             f"commit {meta['commit']} · {meta['host']}", ""]
    if all(r["source"] == "synthetic" for r in results):
        lines += ["> **Synthetic suite only.** These numbers check that the pipeline works and is reproducible. "
                  "They are not evidence of real-world clip quality: the events were generated by the same author who "
                  "wrote the detectors. Add labelled real recordings (bench/README.md) before drawing conclusions.", ""]

    def table(rows_by_sys, title):
        out = [f"## {title}", "", "| system | " + " | ".join(c for c, _ in cols) + " |",
               "|---|" + "---|" * len(cols)]
        for s in systems:
            rs = [r for r in rows_by_sys if s in r["systems"] and "error" not in r["systems"][s]]
            if not rs:
                continue
            ag = metrics.aggregate([r["systems"][s] for r in rs])
            out.append(f"| {s} | " + " | ".join(fmt(ag.get(c), p) for c, p in cols) + " |")
        return out + [""]
    lines += table(results, "All cases (mean per case)")
    for mode in sorted({r["mode"] for r in results}):
        lines += table([r for r in results if r["mode"] == mode], f"Mode: {mode}")
    lines += ["## Per case", "", "| case | " + " | ".join(f"{s} AP / R@R" for s in systems) + " |", "|---|" + "---|" * len(systems)]
    for r in results:
        cells = []
        for s in systems:
            m = r["systems"].get(s, {})
            cells.append("ERROR" if "error" in m else f"{fmt(m.get('ap'))} / {fmt(m.get('r@R'), True)}")
        lines.append(f"| {r['case']} | " + " | ".join(cells) + " |")
    errs = [(r["case"], s, m["error"]) for r in results for s, m in r["systems"].items() if "error" in m]
    if errs:
        lines += ["", "## Errors", ""] + [f"- {c} / {s}: `{e.strip().splitlines()[-1][:200]}`" for c, s, e in errs]
    lines += ["", "Metric definitions: bench/README.md."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(HERE / "cases")); ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--top", type=int, default=8); ap.add_argument("--systems", default="baseline,new,new_novisual")
    ap.add_argument("--whisper-model", default="auto")
    a = ap.parse_args(argv)
    systems = [s for s in a.systems.split(",") if s]
    cases = sorted(p.parent for p in Path(a.cases).glob("*/labels.json"))
    if not cases:
        sys.exit(f"no cases in {a.cases} (run bench/make_synthetic.py, or add labelled recordings)")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = Path(a.out) / stamp
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for c in cases:
        r = run_case(c, systems, out, a.top, a.whisper_model)
        results.append(r)
        print(c.name, " ".join(f"{s}: AP {fmt(m.get('ap'))} R@R {fmt(m.get('r@R'), True)} {fmt(m.get('wall_s'))}s"
                               if "error" not in m else f"{s}: ERROR" for s, m in r["systems"].items()), flush=True)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    from highlight import accel
    import platform
    meta = {"when": stamp, "top": a.top, "commit": commit or "?", "host": f"{platform.system()} {platform.machine()}, "
            f"{os.cpu_count()} CPUs, CUDA devices: {accel.cuda_devices()}"}
    (out / "results.json").write_text(json.dumps({"meta": meta, "results": results}, indent=1), encoding="utf-8")
    (out / "report.md").write_text(report(results, systems, meta), encoding="utf-8")
    print(f"\nreport: {out / 'report.md'}")
    return out


if __name__ == "__main__":
    main()
