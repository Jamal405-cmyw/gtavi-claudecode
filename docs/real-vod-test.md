# Testing on a real GTA livestream VOD (Windows, RTX 4070 SUPER)

Goal: find out whether the new engine picks better clips than the original on real footage, and what
it costs in time, CPU and GPU. Use a VOD you are allowed to clip (a campaign creator's, or your own).

## 0. One-time setup (PowerShell, in the repo folder)

```
git checkout feature/ai-highlight-engine-v1
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest                      # should report all tests passed
python -c "import ctranslate2; print('CUDA devices:', ctranslate2.get_cuda_device_count())"
```

The last line must print `CUDA devices: 1` for GPU transcription. If it prints 0, faster-whisper
falls back to CPU (`small` model) and says so; install the CUDA 12 cuBLAS/cuDNN runtime per the
faster-whisper README to fix it.

## 1. Get the chat (strongly recommended)

- YouTube VOD of a past live stream: nothing to do, chat replay is fetched automatically.
- Twitch VOD: download chat with TwitchDownloaderCLI:
  `TwitchDownloaderCLI chatdownload --id <vod id> -o chat.json`

## 2. Run both detectors blind

```
python bench/compare_vod.py run "https://www.twitch.tv/videos/<id>" --chat chat.json --top 5 --name stream1
```

This downloads once, transcribes once (Whisper large-v3-turbo on the GPU), runs the original detector
and the new engine on identical inputs, measures each one (wall time, CPU seconds, peak RAM, peak GPU
utilisation/VRAM via nvidia-smi), pools their top 5 picks and renders a 480p preview of each.
Long VODs: `--max-minutes 180` (default) limits the download.

Optional VLM judge as a third contender (needs `ollama pull qwen2.5vl:7b`):
`--systems baseline,new,new_judge`

## 3. Rate without knowing which system picked what

Open `bench/vod/stream1/rate.html` in Chrome. For each clip choose Post / Maybe / No (and stars if
you like). Judge the moment *and* its start/end. Do not open `pool.json` first: it reveals the
systems. Click **Download ratings** and save `ratings.json` into `bench/vod/stream1/`.

## 4. Score

```
python bench/compare_vod.py score bench/vod/stream1 --labels
```

`report.md` shows, per system: precision@5 (Post = 1, Maybe = 0.5), post rate, mean stars, Post clips
only that system found, plus wall time, CPU, RAM and GPU numbers. `--labels` saves your Post clips as
`labels.json`, so `python bench/run_benchmark.py --cases bench/vod` re-checks this VOD after every
future change (regression test on real footage).

## Reading the result honestly

- One VOD and one rater is an anecdote. Aim for 5+ VODs from different streamers before deciding.
- Pooled labels only contain moments at least one system found; true recall is unknown.
- If the new engine wins on precision but ranks a clip "No" that the original got right, look at its
  `why:` line in `runs/new/jobs/*/highlights.md` - that tells you which signal misfired.
- Report GPU numbers only from these runs, not from the synthetic benchmark.
