import os
import torch
from pathlib import Path

# Dependency Check
try:
    from boxmot import BotSort
except ImportError:
    raise ImportError("The 'boxmot' library is missing. Please install it via 'pip install boxmot'.")

class SoccerTracker:
    """
    Wrapper class for the BoTSORT tracking algorithm, specifically configured for 
    soccer player tracking. It integrates motion compensation (GMC) and 
    Re-Identification (ReID) features to handle occlusions and fast movements.
    """

    def __init__(self, reid_weights_path, device=None):
        """
        Initializes the tracker with the hyperparameter configuration 
        derived from the optimization phase.

        Args:
            reid_weights_path (str): Path to the fine-tuned OSNet weights.
            device (str, optional): Computation device ('cpu' or 'cuda'). 
                                    If None, it is automatically selected.
        """
        # 1. Device Configuration
        self.device = device if device else ('0' if torch.cuda.is_available() else 'cpu')
        self.half = True if self.device != 'cpu' else False  # Enable FP16 on GPU
        
        # 2. ReID Weights Verification
        if not os.path.exists(reid_weights_path):
            print(f"[TRACKER WARNING] ReID weights file not found at: {reid_weights_path}")
            print("[TRACKER WARNING] The tracker may fallback to default weights or fail.")
        
        # 3. BoTSORT Initialization 
        # We explicitly pass parameters here to override any library defaults 
        # and ensure consistency with the experimental results.
        self.bot_sort = BotSort(
            reid_weights=Path(reid_weights_path),
            device=self.device,
            half=self.half,
            
            # Detection Threshold 
            det_thresh=0.25,
            
            # --- Tracking Hyperparameters ---
            # Thresholds for track activation and maintenance
            track_high_thresh=0.40,   # Confidence to start a new track
            track_low_thresh=0.10,    # Confidence to keep a track alive
            new_track_thresh=0.15,    # Confirmation threshold
            
            # Memory and Association
            track_buffer=250,         # Retain tracks for 10 seconds (250 frames) to handle long occlusions            
            match_thresh=0.95,        # High strictness for ReID matching to prevent ID switches
            
            # Similarity Thresholds
            proximity_thresh=0.70,
            appearance_thresh=0.90,   # ReID similarity threshold
            
            # Camera Motion Compensation and System Settings
            # NOTE: these are the settings that produced the contest results. The Colab run
            # created BoT-SORT without passing `cmc_method` / `fuse_first_associate`, so the
            # BoxMOT defaults were used (ECC camera-motion compensation, no ReID fusion in the
            # first association). They are made explicit here for reproducibility.
            cmc_method='ecc',           # ECC-based global camera motion compensation
            frame_rate=25,              # Standard broadcast frame rate
            with_reid=True,             # Enable deep ReID features (fine-tuned OSNet)
            fuse_first_associate=False  # First association stage uses motion/IoU cues only
        )

    def update(self, detections, orig_image):
        """
        Updates the tracker with new detections from the current frame.

        Args:
            detections (np.ndarray): Array of detections [x1, y1, x2, y2, conf, class].
            orig_image (np.ndarray): The original frame image (needed for ReID feature extraction).

        Returns:
            np.ndarray: Updated tracks [x1, y1, x2, y2, id, conf, class, ind].
        """
        return self.bot_sort.update(detections, orig_image)