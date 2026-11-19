"""Build the synthetic labelled benchmark suite (deterministic: same seed -> same files).

usage: python bench/make_synthetic.py [--out bench/cases] [--n 9] [--seed 100] [--dur 300]

Case mix (cycled): full signals; no chat log; no transcript. Every case has 2-4 labelled events
(laughter, quiet chuckle, shouting, explosion) and 2-3 decoys (loud no-reaction gunfight, music bed,
mic bump, chat spam bot).
See bench/README.md for why this suite cannot stand in for real labelled recordings.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import synth

MODES = ("full", "no_chat", "no_transcript")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "cases"))
    ap.add_argument("--n", type=int, default=9); ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--dur", type=int, default=300)
    a = ap.parse_args(argv)
    out = Path(a.out)
    for i in range(a.n):
        seed = a.seed + i
        mode = MODES[i % len(MODES)]
        name = f"synth_{seed}_{mode}"
        d = synth.build_case(out / name, seed=seed, dur=a.dur, n_events=2 + i % 3, n_decoys=2 + i % 2)
        lab = json.loads((d / "labels.json").read_text())
        lab["source"] = "synthetic"; lab["mode"] = mode
        if mode == "no_chat":
            (d / "chat.json").unlink(); lab.pop("chat", None)
        if mode == "no_transcript":
            (d / "transcript.json").write_text("[]", encoding="utf-8")
        (d / "audio_src.wav").unlink()
        (d / "labels.json").write_text(json.dumps(lab, indent=1), encoding="utf-8")
        print(f"{name}: {len(lab['highlights'])} labelled highlights")


if __name__ == "__main__":
    main()
