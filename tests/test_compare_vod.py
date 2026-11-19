"""Blind real-VOD comparison workflow, exercised on a synthetic stand-in VOD."""
import json, shutil
from conftest import needs_ffmpeg
import compare_vod, synth


def test_pool_merges_only_near_identical_clips():
    picks = {"a": [{"start": 10, "end": 30}, {"start": 100, "end": 120}],
             "b": [{"start": 11, "end": 31}, {"start": 95, "end": 125}]}
    items = compare_vod.pool(picks)
    assert len(items) == 3
    assert next(it for it in items if it["start"] == 10)["by"] == {"a": 1, "b": 1}


def test_score_precision_and_unique_finds():
    items = [{"id": "c1", "start": 0, "end": 20, "by": {"a": 1, "b": 2}},
             {"id": "c2", "start": 50, "end": 70, "by": {"a": 2}},
             {"id": "c3", "start": 90, "end": 110, "by": {"b": 1}}]
    ratings = {"c1": {"post": "yes", "stars": 5}, "c2": {"post": "maybe", "stars": 3}, "c3": {"post": "yes"}}
    sc = compare_vod.score(items, ratings, ["a", "b"], top=2)
    assert sc["a"]["precision@K"] == 0.75 and sc["a"]["mean_stars"] == 4.0 and sc["a"]["post_only_this_system"] == 0
    assert sc["b"]["precision@K"] == 1.0 and sc["b"]["post_only_this_system"] == 1


@needs_ffmpeg
def test_run_and_score_end_to_end(tmp_path):
    case = synth.build_case(tmp_path / "src", seed=31, dur=150, n_events=2, n_decoys=1)
    out = tmp_path / "vod"
    (out / "demo").mkdir(parents=True)
    shutil.copy(case / "transcript.json", out / "demo" / "transcript.json")  # no Whisper in CI
    d = compare_vod.main(["run", str(case / "source.mp4"), "--chat", str(case / "chat.json"), "--top", "3",
                          "--name", "demo", "--out", str(out), "--cam", "0.7:0.05:0.28:0.3"])
    pool = json.loads((d / "pool.json").read_text())
    assert set(pool["systems"]) == {"baseline", "new"} and pool["items"]
    page = (d / "rate.html").read_text()
    assert all((d / "previews" / f"{it['id']}.mp4").exists() and it["id"] in page for it in pool["items"])
    assert "baseline" not in page and '"new"' not in page          # blind: system names never shown
    assert all(pool["runs"][s]["wall_s"] > 0 for s in ("baseline", "new"))
    ratings = {it["id"]: {"post": "yes" if "new" in it["by"] else "no", "stars": 3} for it in pool["items"]}
    (d / "ratings.json").write_text(json.dumps(ratings))
    sc = compare_vod.main(["score", str(d), "--labels"])
    assert sc["new"]["post_rate"] == 1.0
    lab = json.loads((d / "labels.json").read_text())
    assert lab["highlights"] and "pooled" in lab["note"]
    assert "precision@K" in (d / "report.md").read_text()


def test_rating_page_handlers_are_valid_js(tmp_path):
    """Regression: onclick handlers once embedded double-quoted JSON inside a double-quoted attribute."""
    import html.parser, re, shutil as sh, subprocess
    import pytest
    items = [{"id": "clip_01", "start": 0, "end": 20, "by": {"a": 1}}]
    compare_vod.write_page(tmp_path, items, 'Title with "quotes" & <tags>')
    page = (tmp_path / "rate.html").read_text()
    handlers = []

    class P(html.parser.HTMLParser):
        def handle_starttag(self, tag, attrs):
            handlers.extend(v for k, v in attrs if k in ("onclick", "oninput"))
    P().feed(page)
    handlers = [h for h in handlers if h != "dl()"]
    assert len(handlers) == 9 and all(h.startswith("set(") and h.endswith(")") for h in handlers)
    if not sh.which("node"):
        pytest.skip("node not installed")
    script = re.search(r"<script>(.*)</script>", page, re.S).group(1)
    js = tmp_path / "t.js"
    js.write_text("var localStorage={getItem:()=>null,setItem:()=>{}};var document={querySelectorAll:()=>[],"
                  "getElementById:()=>({})};\n" + script.replace("paint();", "") +
                  "\nvar out=[];function paint(){};\n" + "\n".join(h.replace("this.value", "'x'") + ";" for h in handlers)
                  + "\nif(R['clip_01'].post!=='no'||R['clip_01'].stars!==5)throw new Error('bad '+JSON.stringify(R));")
    r = subprocess.run(["node", str(js)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
