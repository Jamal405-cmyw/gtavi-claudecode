"""Facecam-aware vertical reframing (reframe.py + cut.py --reframe)."""
import json, os, subprocess, sys
import numpy as np
import pytest
from conftest import ROOT, needs_ffmpeg
import reframe


def _frames(n, h=90, w=160, seed=0):
    rng = np.random.default_rng(seed)
    base = (rng.random((h, w)) * 60 + 60).astype(np.uint8)
    return np.stack([np.roll(base, i, axis=1) for i in range(n)])


def test_scene_cut_rule_matches_adaptive_detector():
    fr = _frames(20)
    fr[10:] = 255 - fr[10:]                                  # hard cut at frame 10
    assert reframe.scene_cuts(fr, fps=4) == [2.5]
    assert reframe.scene_cuts(_frames(20), fps=4) == []      # steady panning is not a cut


def test_centre_by_default_and_webcam_never_in_crop():
    cols = np.ones(160)
    assert reframe._best_center(cols, 0.5) == pytest.approx(0.5, abs=0.01)
    cam = (0.62, 0.05, 0.33, 0.30)                             # webcam overlapping a centred crop
    c = reframe._best_center(cols, 0.5, cam=cam)
    assert c + 0.25 <= 0.62 + 1e-6                             # crop ends before the webcam starts
    assert reframe._best_center(cols, 1.0, cam=cam) == 0.5     # full-width crop: nothing to choose


def test_motion_nudge_is_capped_and_uniform_motion_stays_centred():
    cols = np.zeros(160); cols[115:140] = 100.0                # strong action right of centre (0.72-0.87)
    c = reframe._best_center(cols, 0.4)
    assert 0.5 < c <= 0.65 + 1e-6                              # follows it, but at most 0.15 off centre
    assert reframe._best_center(np.full(160, 50.0), 0.4) == pytest.approx(0.5, abs=0.01)  # camera pan everywhere


def test_plan_is_piecewise_static_per_scene():
    fr = np.full((40, 90, 160), 90, np.uint8)
    fr[20:] = 170                                              # hard cut at 5 s (scene 2 is brighter)
    for i in range(40):                                        # a moving car: right side, then left side
        y = (i * 7) % 70
        x0 = 125 if i < 20 else 15
        fr[i, y:y + 20, x0:x0 + 20] = 250
    p = reframe.plan_game_crop(fr, 4, 0.45)
    assert p["cuts"] == [5.0] and len(p["segments"]) == 2
    a, b = p["segments"]
    assert a["cx"] > 0.5 > b["cx"] and a["t1"] == b["t0"] == 5.0
    assert abs(a["cx"] - 0.5) <= 0.15 + 1e-6 and abs(b["cx"] - 0.5) <= 0.15 + 1e-6


def test_short_segments_are_merged():
    fr = _frames(40)
    for k in (8, 10, 30):                                      # cuts 0.5 s apart -> tiny segments
        fr[k:] = 255 - fr[k:]
    p = reframe.plan_game_crop(fr, 4, 0.45)
    assert all(s["t1"] - s["t0"] >= 1.5 for s in p["segments"])


def test_face_crop_centres_on_the_face_inside_the_box():
    cam = (0.60, 0.05, 0.38, 0.38)                              # wide 16:9-ish webcam
    face = (0.86, 0.12, 0.06, 0.10)                             # streamer sits at the right of their cam
    x, y, w, h = reframe.face_crop(cam, face, 1920, 1080, 1080, 729)
    assert x >= cam[0] - 1e-9 and x + w <= cam[0] + cam[2] + 1e-9 and y >= cam[1] - 1e-9
    assert (w * 1920) / (h * 1080) == pytest.approx(1080 / 729, rel=1e-3)
    fcx = face[0] + face[2] / 2
    assert x < fcx < x + w and abs((x + w / 2) - fcx) < abs((cam[0] + cam[2] / 2) - fcx)


def test_crop_expression_quoting():
    e = reframe.crop_x_expr([{"t0": 0, "t1": 3.2, "cx": 0.4}, {"t0": 3.2, "t1": 9, "cx": 0.6}])
    assert e == "clip(iw*if(lt(t\\,3.200)\\,0.4000\\,0.6000)-ow/2\\,0\\,iw-ow)"


def _scene_video(path, dur=16, W=640, H=360, cam=(0.72, 0.04, 0.26, 0.30)):
    """Gray world, a red 'player' at the centre, a white-framed bright webcam top-right."""
    fr = np.full((H, W, 3), 70, np.uint8)
    fr[int(H * 0.45):int(H * 0.75), int(W * 0.47):int(W * 0.53)] = (0, 0, 230)   # BGR red player
    x0, y0, cw, ch = int(cam[0] * W), int(cam[1] * H), int(cam[2] * W), int(cam[3] * H)
    fr[y0:y0 + ch, x0:x0 + cw] = (240, 240, 240)
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", "10",
                          "-i", "-", "-f", "lavfi", "-i", "sine=f=220:d=%d" % dur, "-c:v", "libx264", "-preset", "ultrafast",
                          "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], stdin=subprocess.PIPE)
    rng = np.random.default_rng(0)
    for i in range(dur * 10):
        f = fr.copy()
        f[:int(H * 0.4), :int(W * 0.6)] = np.clip(70 + rng.integers(-8, 8, (int(H * 0.4), int(W * 0.6), 1)), 0, 255)
        p.stdin.write(f.tobytes())
    p.stdin.close(); p.wait()
    return cam


def _cut(job, home, *extra):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "cut.py"), str(job), "--pick", "1", "--captions", "none", *extra],
                       env=dict(os.environ, CLIP_DESK_HOME=str(home)), capture_output=True, text=True, cwd=home)
    assert r.returncode == 0, r.stderr[-800:]
    return job / "edits" / "clip_1.mp4"


def _panel_stats(mp4, y0, y1):
    """Share of red (player) and near-white (webcam) pixels in rows y0..y1 of the middle frame."""
    import cv2
    out = mp4.with_suffix(".png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "5", "-i", str(mp4), "-frames:v", "1", str(out)], check=True)
    img = cv2.imread(str(out))[y0:y1]
    red = ((img[:, :, 2] > 150) & (img[:, :, 1] < 80) & (img[:, :, 0] < 80)).mean()
    white = (img.min(axis=2) > 200).mean()
    return red, white


@needs_ffmpeg
@pytest.mark.parametrize("layout,top", [("react", int(1920 * 0.38)), ("facecam", int(1920 * 0.38))])
def test_cut_reframe_keeps_player_and_drops_webcam(tmp_path, layout, top):
    src = tmp_path / "src.mp4"
    cam = _scene_video(src)
    job = tmp_path / "job"; job.mkdir()
    (job / "transcript.json").write_text("[]")
    (job / "candidates.json").write_text(json.dumps({"source": str(src), "info": {}, "duration": 16,
                                                     "candidates": [{"n": 1, "start": 0.0, "end": 15.0, "score": 1, "why": [], "text": ""}]}))
    cams = ":".join(map(str, cam))
    new = _cut(job, tmp_path, "--layout", layout, "--cam", cams)
    meta = json.loads((job / "edits" / "clip_1.json").read_text())
    assert meta["reframe"]["segments"] and meta["reframe"]["crop_w"] < 1
    red_new, white_new = _panel_stats(new, top + 10, 1920)
    old = _cut(job, tmp_path, "--layout", layout, "--cam", cams, "--reframe", "off")
    red_old, white_old = _panel_stats(old, top + 10, 1920)
    ph = 1920 - top                                             # panel height; source scaled to it
    full_red = (0.30 * ph) * (0.06 * 640 * ph / 360) / ((1920 - top - 10) * 1080)  # whole player visible
    assert white_new < 0.002                                    # webcam not duplicated in the gameplay panel
    assert red_new > 0.9 * full_red                             # the centred player is fully in frame
    if layout == "react":
        assert red_old < 0.8 * full_red                         # old fixed edge crop cut part of the player off


@pytest.mark.parametrize("layout", ["fill", "blur", "react", "facecam"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_old_call_signature_is_byte_identical(layout, side):
    """translate.py and --reframe off call video_filter(layout, cam, ass[, band, cam_side]); output must not change."""
    import importlib.util
    import cut
    spec = importlib.util.spec_from_file_location("cut_before", ROOT / "tests" / "fixtures" / "cut_before_reframe.py")
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    cam = "0.7000:0.0500:0.2800:0.3000"
    assert cut.video_filter(layout, cam, "/tmp/a.ass", cam_side=side) == old.video_filter(layout, cam, "/tmp/a.ass", cam_side=side)
    band = (600, 200, 3)
    assert cut.video_filter(layout, cam, "/tmp/a.ass", band=band, cam_side=side) == \
        old.video_filter(layout, cam, "/tmp/a.ass", band=band, cam_side=side)
