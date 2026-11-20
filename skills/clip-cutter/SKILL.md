---
name: clip-cutter
description: Cut a picked moment into a finished 1080x1920 vertical clip with word-by-word captions, a hook box and a credit line. Use after moment-finder, or when the user says "cut it", "make the clip", "make it vertical".
---

# Clip cutter

Run:

`python "${CLAUDE_PLUGIN_ROOT}/scripts/cut.py" jobs/<id> --pick <n> --layout fill|blur|facecam --hook "TEXT" --credit "@creator"`

- `fill`: crop the game video to fill the frame. Default for gameplay.
- `blur`: full video centered over a blurred copy. Use when cropping would cut off the action.
- `facecam`: gameplay with the streamer's camera shown. Use for reactions.

Reframing is automatic (`--reframe auto`): the gameplay panel stays on the player (screen centre),
follows the action within limits, switches position only at scene cuts, and never shows the webcam a
second time; the camera panel is centred on the streamer's face when it was detected. Use
`--reframe off` to get the previous fixed crops.

Before running:

1. Get 3 hook options from the `hook-writer` skill and use the best one.
2. Set `--credit` to the original creator's handle. For official material use `@RockstarGames`. Never leave it blank.

After running:

1. Output is an mp4 in `media/edits`. Check it is 1080x1920, the hook shows in the first 2 seconds, captions are readable, the credit line is present.
2. Run `rule-check` on it. Fix and re-cut until it passes.
3. Trim 7 to 30 seconds for most clips. End on the payoff.
4. Show the human the file, the caption text and the credit, and ask for "yes". Do not upload anything.
