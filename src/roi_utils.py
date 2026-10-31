"""Geometry helpers for the behavior-analysis task (player counting inside ROIs)."""

import json
import os
from typing import Dict, Optional, Tuple

Rect = Tuple[int, int, int, int]


def load_rois(roi_json_path: str) -> Optional[Dict[str, dict]]:
    """Loads normalized ROIs ({"roi1": {"x", "y", "width", "height"}, ...}) from a JSON file."""
    if not os.path.exists(roi_json_path):
        return None
    with open(roi_json_path, "r") as f:
        return json.load(f)


def roi_to_pixel_rect(roi: dict, img_w: int, img_h: int) -> Rect:
    """Converts a normalized ROI into a pixel rectangle (x1, y1, x2, y2)."""
    x1 = int(round(roi["x"] * img_w))
    y1 = int(round(roi["y"] * img_h))
    x2 = x1 + int(round(roi["width"] * img_w))
    y2 = y1 + int(round(roi["height"] * img_h))
    return x1, y1, x2, y2


def bbox_ground_point(x: float, y: float, w: float, h: float) -> Tuple[int, int]:
    """Bottom-center of a bounding box: approximates the player's position on the pitch."""
    return int(round(x + w / 2.0)), int(round(y + h))


def point_in_rect(px: int, py: int, rect: Rect) -> bool:
    """True if the point lies inside the rectangle (borders included)."""
    x1, y1, x2, y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2
