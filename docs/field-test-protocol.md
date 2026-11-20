# Field test protocol: one real GTA Twitch VOD, three configurations, blind review

Status: **prepared, not run.** Nothing below has been executed on the target PC. Each stage needs the
owner's approval before it runs.

## Configurations

| | Detector | Editing | Ranking |
|---|---|---|---|
| A | original `moments.py` (frozen copy, `bench/baseline/`) | original `cut.py` (frozen copy) | original |
| B | `highlights.py` v1.2 (speech-aware edges) | `cut.py --reframe auto` | engine score |
| C | same as B | same as B | + local VLM (`--judge ollama`, qwen2.5vl:7b) |

Held constant: the same downloaded video file, one Whisper transcript (made once, copied to every
configuration), one chat file, 5 clips per configuration, layout `react` (facecam on top, gameplay
below), word captions, credit line, 1080×1920, −14 LUFS. B and C share the detector, so any span they
both pick is rendered once and reviewed once.

## Rules

1. Baseline first: no detector or editing change between `prepare` and `score`. The harness hashes
   the code at `prepare`; `score` flags any change in the report.
2. The only fixes allowed during the test are fixes for crashes. Each is recorded, and the affected
   configuration is rerun from `detect`. No threshold or weight changes based on this VOD.
3. Rate on the blind page before opening `key.json`, `picks.json`, `renders.json` or `report.md`.
4. Nothing is published or uploaded. Outputs stay in `bench\field\<name>\` (git-ignored).

## Stages and approvals

| Stage | Command (PowerShell, in `C:\Users\jamal\gtavi-test`) | Changes / downloads | GPU |
|---|---|---|---|
| 0 read-only checks | `Get-PSDrive C`, `nvidia-smi`, `where.exe TwitchDownloaderCLI ollama` | none | none |
| 1 setup | clone branch, `py -3.14 -m venv --system-site-packages .venv`, `pip install opencv-python-headless "yt-dlp[default]" psutil pytest`, `pytest -q` | new folder; ~60 MB packages into the venv only | none |
| 2 tools | install TwitchDownloaderCLI (GitHub release, x64 Windows); install Ollama and `ollama pull qwen2.5vl:7b` | CLI zip; Ollama app; 6.0 GB model | none |
| 3 plan | `python bench\field_test.py plan "<VOD URL>" --minutes 90` | none (metadata only) | none |
| 4 prepare | `python bench\field_test.py prepare "<VOD URL>" --name vod1 --minutes 90` | VOD (size from step 3), chat, Whisper large-v3-turbo 1.62 GB (first time) | Whisper fp16 |
| 5 detect | `python bench\field_test.py detect vod1` | none | light NVDEC frame sampling; config C loads the VLM |
| 6 render | `python bench\field_test.py render vod1 --credit "@streamer"` | ≤ 15 MP4 files | none (libx264 on CPU) |
| 7 page | `python bench\field_test.py page vod1` → open `bench\field\vod1\blind\rate.html` | copies of the renders | none |
| 8 score | `python bench\field_test.py score vod1` (after saving `ratings.json` in `blind\`) | report files | none |

`python` = `.\.venv\Scripts\python.exe`.

## Outputs

`report.md` with three separate sections: (1) highlight selection, (2) editing quality, (3)
processing time / CPU / RAM / GPU utilisation / GPU memory per step and every failure. Per-clip
table with original VOD timestamps. `results.json`, `log.json` for the raw numbers.

## Known limits of this design

- One VOD, one rater: an anecdote. Repeat on more VODs before drawing conclusions.
- Blinding is imperfect: the original detector's clips tend to be ~40 s and its framing differs.
- Subtitles come from the same transcript in all configurations; differences are boundary effects.
- GPU numbers are peak samples every 0.5 s from `nvidia-smi`, whole GPU (other apps count too).
