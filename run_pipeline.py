"""
End-to-end inference pipeline: detection (YOLO11l) -> tracking (BoT-SORT + OSNet ReID)
-> behavior analysis (players per ROI) -> offline trajectory post-processing.

Expected dataset layout (one folder per video sequence):

    <data-dir>/
        <video_id>/
            img1/000001.jpg, 000002.jpg, ...
            roi.json                      # optional, needed for behavior analysis

Outputs (one pair of files per video):

    <output-dir>/tracking_<video_id>_<team_id>.txt   # frame,id,x,y,w,h
    <output-dir>/behavior_<video_id>_<team_id>.txt   # frame,roi_id,count

Usage:
    python run_pipeline.py --data-dir data/test --output-dir predictions
"""

import argparse
import os
import shutil
import sys
import tempfile

# Avoids the "OMP: Error #15" crash when PyTorch and OpenCV ship different OpenMP runtimes
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

from src.post_processing import SoccerTrackCleaner
from src.roi_utils import bbox_ground_point, load_rois, point_in_rect, roi_to_pixel_rect
from src.tracker import SoccerTracker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Detector inference hyperparameters
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7
IMG_SIZE = 1920  # native resolution: critical for small, distant players

# Post-processing hyperparameters (tuned on the validation split)
CLEANER_CONFIG = {
    "do_stitching": True, "max_stitch_gap": 150, "max_stitch_dist": 200,
    "do_interpolation": True, "max_interp_gap": 15,
    "do_smoothing": True, "smooth_window": 15, "smooth_poly": 2,
    "do_cleanup": True, "min_track_len": 50,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Soccer player tracking & behavior analysis pipeline")
    parser.add_argument("--data-dir", default=os.path.join(BASE_DIR, "data", "test"),
                        help="Folder containing one sub-folder per video sequence")
    parser.add_argument("--output-dir", default=os.path.join(BASE_DIR, "predictions"),
                        help="Where tracking/behavior files are written")
    parser.add_argument("--weights-dir", default=os.path.join(BASE_DIR, "weights"),
                        help="Folder with yolov11l_finetuned_best.pt and osnet_x1_0_finetuned_best.pt")
    parser.add_argument("--team-id", default="23", help="Suffix used in output file names")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run the contest evaluator (requires the course-provided evaluation/simulator.py)")
    return parser.parse_args()


def process_video(model, reid_path, video_dir, track_file, behav_file, device):
    """Runs detection + tracking on one sequence and writes raw tracking and behavior files."""
    rois = load_rois(os.path.join(video_dir, "roi.json"))
    tracker = SoccerTracker(reid_weights_path=reid_path)  # fresh tracker: resets IDs and history

    results = model.predict(source=os.path.join(video_dir, "img1"), conf=CONF_THRESHOLD,
                            iou=IOU_THRESHOLD, imgsz=IMG_SIZE, stream=True, verbose=False,
                            classes=[0], device=device)

    with open(track_file, "w") as f_track, open(behav_file, "w") as f_behav:
        for result in results:
            frame_id = int(os.path.splitext(os.path.basename(result.path))[0])
            h_img, w_img = result.orig_shape

            dets = result.boxes.data.cpu().numpy()
            if len(dets) == 0:
                dets = np.empty((0, 6))
            tracks = tracker.update(dets, result.orig_img)

            roi_rects = {name: roi_to_pixel_rect(r, w_img, h_img) for name, r in (rois or {}).items()}
            roi_counts = {1: 0, 2: 0}

            for track in tracks:
                x1, y1, x2, y2 = track[:4]
                tid = int(track[4])
                w, h = x2 - x1, y2 - y1
                f_track.write(f"{frame_id},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f}\n")

                # A player is inside an ROI if the bottom-center of its box (its feet) is inside
                px, py = bbox_ground_point(x1, y1, w, h)
                for roi_id in (1, 2):
                    rect = roi_rects.get(f"roi{roi_id}")
                    if rect and point_in_rect(px, py, rect):
                        roi_counts[roi_id] += 1

            # Note: counts are computed on the online tracks, before offline post-processing
            f_behav.write(f"{frame_id},1,{roi_counts[1]}\n")
            f_behav.write(f"{frame_id},2,{roi_counts[2]}\n")


def run_evaluation(data_dir, output_dir, team_id):
    """Optional: official contest metrics (HOTA@0.5, nMAE). The evaluator is not part of this repo."""
    sys.path.insert(0, os.path.join(BASE_DIR, "evaluation"))
    try:
        from simulator import simulate_and_eval_hota05
    except ImportError:
        print("[EVAL] evaluation/simulator.py not found: skipping evaluation (see README).")
        return
    simulate_and_eval_hota05(dataset_root=data_dir, predictions_root=output_dir, group=team_id,
                             fps=25.0, split="test", show_window=False)


def main():
    args = parse_args()
    detector_path = os.path.join(args.weights_dir, "yolov11l_finetuned_best.pt")
    reid_path = os.path.join(args.weights_dir, "osnet_x1_0_finetuned_best.pt")

    for path, what in [(detector_path, "detector weights"), (reid_path, "ReID weights"),
                       (args.data_dir, "data directory")]:
        if not os.path.exists(path):
            sys.exit(f"[ERROR] Missing {what}: {path}")

    videos = sorted(d for d in os.listdir(args.data_dir)
                    if os.path.isdir(os.path.join(args.data_dir, d, "img1")))
    if not videos:
        sys.exit(f"[ERROR] No video sequences (folders with an 'img1' sub-folder) in {args.data_dir}")

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"[1/3] Loading detector on {'GPU' if device == '0' else 'CPU'}...")
    model = YOLO(detector_path)

    os.makedirs(args.output_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as raw_dir:
        print(f"[2/3] Tracking {len(videos)} video sequences...")
        for vid in tqdm(videos, desc="Inference"):
            process_video(model, reid_path, os.path.join(args.data_dir, vid),
                          os.path.join(raw_dir, f"tracking_{vid}_{args.team_id}.txt"),
                          os.path.join(raw_dir, f"behavior_{vid}_{args.team_id}.txt"), device)

        print("[3/3] Offline post-processing (stitching, interpolation, smoothing, cleanup)...")
        SoccerTrackCleaner(input_dir=raw_dir, output_dir=args.output_dir, **CLEANER_CONFIG).process_all_files()

    print(f"[DONE] Predictions saved in {args.output_dir}")
    if args.evaluate:
        run_evaluation(args.data_dir, args.output_dir, args.team_id)


if __name__ == "__main__":
    main()
