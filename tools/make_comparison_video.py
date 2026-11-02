"""
Renders a side-by-side video (ground truth | predictions) for one sequence, with ROIs,
per-ROI player counts and track IDs. Optionally exports a short GIF for the README.

Usage:
    python tools/make_comparison_video.py --seq-dir data/demo/SNMOT-150 \
        --pred-file predictions/tracking_SNMOT-150_23.txt --out demo.mp4 \
        --gif assets/demo.gif --gif-start 0 --gif-seconds 6
"""

import argparse
import configparser
import glob
import json
import os
import subprocess
import sys
from collections import defaultdict

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.roi_utils import bbox_ground_point, point_in_rect, roi_to_pixel_rect  # noqa: E402

ROI_COLORS = {"roi1": (255, 0, 0), "roi2": (0, 0, 255)}  # BGR: blue, red
OUTSIDE_COLOR = (0, 255, 255)                             # yellow


def read_boxes(path, skip_ids=()):
    """frame -> [(id, x, y, w, h)] from a MOT-style text file."""
    per_frame = defaultdict(list)
    with open(path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 6 or int(float(p[1])) in skip_ids:
                continue
            per_frame[int(float(p[0]))].append((int(float(p[1])), *(float(v) for v in p[2:6])))
    return per_frame


def ball_ids(gameinfo_path):
    """Tracklet IDs labelled as 'ball' in SoccerNet's gameinfo.ini (excluded from the GT view)."""
    if not os.path.exists(gameinfo_path):
        return set()
    cfg = configparser.ConfigParser()
    cfg.read(gameinfo_path)
    sec = cfg["Sequence"]
    return {int(k.split("_")[1]) for k, v in sec.items() if k.startswith("trackletid_") and "ball" in v}


def draw_panel(img, boxes, rois, title):
    h, w = img.shape[:2]
    rects = {name: roi_to_pixel_rect(r, w, h) for name, r in rois.items()}
    counts = {name: 0 for name in rects}

    for name, (x1, y1, x2, y2) in rects.items():
        cv2.rectangle(img, (x1, y1), (x2, y2), ROI_COLORS.get(name, (0, 255, 0)), 2)

    for tid, x, y, bw, bh in boxes:
        px, py = bbox_ground_point(x, y, bw, bh)
        inside = next((n for n, r in rects.items() if point_in_rect(px, py, r)), None)
        if inside:
            counts[inside] += 1
        color = ROI_COLORS.get(inside, OUTSIDE_COLOR)
        p1, p2 = (int(x), int(y)), (int(x + bw), int(y + bh))
        cv2.rectangle(img, p1, p2, color, 2)
        cv2.putText(img, str(tid), (p1[0], max(12, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    cv2.rectangle(img, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(img, title, (15, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    for name, (x1, y1, _, _) in rects.items():
        cv2.putText(img, f"{name}: {counts[name]}", (x1 + 10, max(y1, 50) + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    ROI_COLORS.get(name, (0, 255, 0)), 3, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", required=True, help="Sequence folder with img1/, gt/gt.txt, roi.json")
    ap.add_argument("--pred-file", required=True, help="tracking_<id>_<team>.txt produced by run_pipeline.py")
    ap.add_argument("--out", default="comparison.mp4")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--gif", help="Optional GIF output path")
    ap.add_argument("--gif-start", type=float, default=0.0)
    ap.add_argument("--gif-seconds", type=float, default=6.0)
    ap.add_argument("--gif-width", type=int, default=960)
    args = ap.parse_args()

    frames = sorted(glob.glob(os.path.join(args.seq_dir, "img1", "*.jpg")))
    if not frames:
        sys.exit(f"No frames in {args.seq_dir}/img1")
    with open(os.path.join(args.seq_dir, "roi.json")) as f:
        rois = json.load(f)
    gt_path = os.path.join(args.seq_dir, "gt", "gt.txt")
    gt = read_boxes(gt_path, ball_ids(os.path.join(args.seq_dir, "gameinfo.ini"))) if os.path.exists(gt_path) else None
    pred = read_boxes(args.pred_file)

    h, w = cv2.imread(frames[0]).shape[:2]
    out_w = w * 2 if gt is not None else w
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (out_w, h))

    for path in frames:
        fid = int(os.path.splitext(os.path.basename(path))[0])
        img = cv2.imread(path)
        right = img.copy()
        draw_panel(right, pred.get(fid, []), rois, "PREDICTION (YOLO11l + BoT-SORT + OSNet + post-proc.)")
        if gt is not None:
            draw_panel(img, gt.get(fid, []), rois, "GROUND TRUTH")
            frame = cv2.hconcat([img, right])
            cv2.line(frame, (w, 0), (w, h), (255, 255, 255), 3)
        else:
            frame = right
        writer.write(frame)
    writer.release()
    print(f"[OK] Video saved: {args.out}")

    if args.gif:
        vf = f"fps=8,scale={args.gif_width}:-1:flags=lanczos"
        pal = args.gif + ".palette.png"
        common = ["ffmpeg", "-v", "error", "-y", "-ss", str(args.gif_start), "-t", str(args.gif_seconds), "-i", args.out]
        subprocess.run(common + ["-vf", f"{vf},palettegen=max_colors=128:stats_mode=diff", pal], check=True)
        subprocess.run(common + ["-i", pal, "-lavfi", f"{vf} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                                 "-loop", "0", args.gif], check=True)
        os.remove(pal)
        print(f"[OK] GIF saved: {args.gif} ({os.path.getsize(args.gif) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
