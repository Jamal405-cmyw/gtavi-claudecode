# GPU transcription check — target PC (2026-11-20)

Measured on the owner's PC in the isolated `C:\Users\jamal\gtavi-test` environment with
`bench/gpu_check.py` (no CPU fallback) and the production path `moments.transcribe`.

## Environment

| Item | Value |
|---|---|
| OS | Windows NT 10.0.26200 |
| Python | 3.14.5 (64-bit), clean venv (`include-system-site-packages = false`) |
| GPU | NVIDIA GeForce RTX 4070 SUPER, 12 282 MiB, compute capability 8.9 |
| Driver | 617.14 (CUDA driver API 13.4) |
| faster-whisper / CTranslate2 | 1.2.1 / 4.9.0 (built for CUDA 12.4) |
| CUDA libraries (venv only) | nvidia-cublas-cu12 12.9.2.10, nvidia-cudnn-cu12 9.28.1.4, nvidia-cuda-nvrtc-cu12 12.9.86 — 1.95 GB installed |
| Model | Whisper large-v3-turbo, float16 (`mobiuslabsgmbh/faster-whisper-large-v3-turbo`, 1.62 GB, downloaded in 23 s) |

No CUDA toolkit or cuDNN was installed system-wide; `scripts/cuda_dlls.py` adds the venv's NVIDIA DLL
folders to the process's DLL search path only.

## Results

Test audio: English speech generated locally with the Windows speech synthesizer from a known
GTA-style script (56.3 s), and the same clip looped to 300 s. Synthetic voice is easier than a live
stream with game audio, so treat speed and accuracy as upper bounds.

| Run | Audio | Model load | Transcription | Real-time factor | GPU memory added | Peak GPU memory (whole card) | WER |
|---|---|---|---|---|---|---|---|
| gpu_check | 56.3 s | 1.80 s | 1.07 s | 52.6× | 2 323 MB | 3 805 MB (idle 1 482 MB from other apps) | 3.3 % |
| gpu_check | 300 s | 1.88 s | 7.22 s | 41.6× | 2 452 MB | 3 924 MB (idle 1 472 MB) | – (looped audio) |
| moments.transcribe (auto device → large-v3-turbo / cuda / float16) | 56.3 s | – | 2.83 s incl. load | – | – | – | – |

GPU memory is sampled from `nvidia-smi` every 0.1 s for the whole card; "added" = peak minus the idle
reading taken just before the model loaded.

## Problems found and fixed during the check

1. No cuBLAS/cuDNN on the PC (the CTranslate2 wheel ships neither) → pip packages in the venv +
   `cuda_dlls.register()` (commit b010506).
2. faster-whisper 1.2.1's file decoder calls `av.open(metadata_errors=...)`, rejected by PyAV 19.
   The pipeline passes samples and is unaffected; `gpu_check` now does the same (commit cdfeff3,
   regression test).
3. Git for Windows CRLF checkouts broke two byte-hash tests (commit 1aa14cf).
