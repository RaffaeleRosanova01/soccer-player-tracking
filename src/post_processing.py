import os
import glob
import shutil
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

class SoccerTrackCleaner:
    """
    Offline post-processing for multi-object tracking results in soccer videos.

    Applies, in order:
      1. Track stitching: merges fragmented tracklets that are close in space and time
         (or that leave and re-enter the frame near the borders).
      2. Linear interpolation: fills short gaps inside each trajectory.
      3. Smoothing: Savitzky-Golay filter on box coordinates to reduce jitter.
      4. Cleanup: drops tracks shorter than ``min_track_len`` frames (likely false positives).

    Input/output files use the MOT-like format: frame, id, x, y, w, h.
    Behavior files are copied unchanged to the output directory.
    """
    def __init__(self, input_dir, output_dir, img_size=(1920, 1080),
                 do_stitching=True, max_stitch_gap=150, max_stitch_dist=200, border_margin=50,
                 do_interpolation=True, max_interp_gap=15,
                 do_smoothing=True, smooth_window=15, smooth_poly=2,
                 do_cleanup=True, min_track_len=50):
        
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.W, self.H = img_size
        
        self.do_stitching = do_stitching
        self.max_stitch_gap = max_stitch_gap
        self.max_stitch_dist = max_stitch_dist
        self.border_margin = border_margin
        
        self.do_interpolation = do_interpolation
        self.max_interp_gap = max_interp_gap
        
        self.do_smoothing = do_smoothing
        self.smooth_window = smooth_window
        self.smooth_poly = smooth_poly
        
        self.do_cleanup = do_cleanup
        self.min_track_len = min_track_len
        
        self.output_cols = ['frame', 'id', 'x', 'y', 'w', 'h']

    def _load_mot_file(self, filepath):
        if not os.path.exists(filepath): return None
        try:
            df = pd.read_csv(filepath, header=None, sep=',')
            if df.empty: return None
            if df.shape[1] >= 6:
                df = df.iloc[:, :6]
                df.columns = ['frame', 'id', 'x', 'y', 'w', 'h']
                for c in ['conf', 'c1', 'c2', 'c3']: df[c] = -1
            return df
        except Exception: return None

    def _is_border(self, cx, cy):
        return (cx < self.border_margin or cx > self.W - self.border_margin or 
                cy < self.border_margin or cy > self.H - self.border_margin)

    def _run_stitching(self, df):
        df = df.sort_values(['frame', 'id'])
        unique_ids = df['id'].unique()
        tracks = {}
        for tid in unique_ids:
            t = df[df['id'] == tid]
            s, e = t.iloc[0], t.iloc[-1]
            tracks[tid] = {
                's_f': s['frame'], 'e_f': e['frame'],
                's_c': (s['x']+s['w']/2, s['y']+s['h']/2),
                'e_c': (e['x']+e['w']/2, e['y']+e['h']/2),
                'avg_w': t['w'].mean()
            }
        
        sorted_tids = sorted(tracks.keys(), key=lambda x: tracks[x]['s_f'])
        id_map = {tid: tid for tid in sorted_tids}
        
        for i in range(len(sorted_tids)):
            curr = sorted_tids[i]
            curr_stats = tracks[curr]
            best_prev, min_dist = None, float('inf')
            
            for j in range(i-1, -1, -1):
                prev = sorted_tids[j]
                while prev != id_map[prev]: prev = id_map[prev]
                if prev == curr: continue
                prev_stats = tracks[prev]
                gap = curr_stats['s_f'] - prev_stats['e_f']
                if gap <= 0 or gap > self.max_stitch_gap: continue
                dist = np.sqrt(np.sum((np.array(curr_stats['s_c']) - np.array(prev_stats['e_c']))**2))
                should_merge = False
                if gap < 25 and dist < (curr_stats['avg_w'] * 3): should_merge = True
                if self._is_border(*curr_stats['s_c']) and self._is_border(*prev_stats['e_c']) and dist < self.max_stitch_dist: should_merge = True
                if should_merge and dist < min_dist:
                    min_dist = dist
                    best_prev = prev
            if best_prev is not None:
                id_map[curr] = best_prev
                tracks[best_prev]['e_f'] = curr_stats['e_f']
                tracks[best_prev]['e_c'] = curr_stats['e_c']

        final_map = {}
        for k in id_map:
            root = k
            while root != id_map[root]: root = id_map[root]
            final_map[k] = root
        df['id'] = df['id'].map(final_map)
        return df

    def _run_interpolation(self, df):
        if not self.do_interpolation: return df
        new_rows = []
        for tid, group in df.groupby('id'):
            group = group.sort_values('frame')
            frames = group['frame'].values
            for i in range(len(frames)-1):
                gap = frames[i+1] - frames[i]
                if 1 < gap <= self.max_interp_gap:
                    r1, r2 = group.iloc[i], group.iloc[i+1]
                    for step in range(1, int(gap)):
                        frac = step / gap
                        row = r1.copy()
                        row['frame'] = int(frames[i] + step)
                        for col in ['x', 'y', 'w', 'h']:
                            row[col] = r1[col] + (r2[col] - r1[col]) * frac
                        new_rows.append(row)
        if new_rows:
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            df = df.sort_values(['frame', 'id'])
        return df

    def _run_smoothing(self, df):
        if not self.do_smoothing: return df
        smoothed = []
        for tid, group in df.groupby('id'):
            group = group.sort_values('frame')
            if len(group) >= self.smooth_window:
                try:
                    for col in ['x', 'y', 'w', 'h']:
                        group[col] = savgol_filter(group[col], self.smooth_window, self.smooth_poly)
                except ValueError:
                    # Window larger than the track or invalid parameters: keep raw values
                    pass
            smoothed.append(group)
        return pd.concat(smoothed) if smoothed else df

    def _run_cleanup(self, df):
        if not self.do_cleanup: return df
        c = df['id'].value_counts()
        return df[df['id'].isin(c[c >= self.min_track_len].index)]

    def process_all_files(self):
        os.makedirs(self.output_dir, exist_ok=True)
        track_files = glob.glob(os.path.join(self.input_dir, "tracking_*.txt"))
        
        print(f"[CLEANER] Processing {len(track_files)} files...")
        
        for fpath in track_files:
            df = self._load_mot_file(fpath)
            if df is None: continue
            
            if self.do_stitching: df = self._run_stitching(df)
            if self.do_interpolation: df = self._run_interpolation(df)
            if self.do_smoothing: df = self._run_smoothing(df)
            if self.do_cleanup: df = self._run_cleanup(df)
            
            df['frame'] = df['frame'].astype(int)
            df['id'] = df['id'].astype(int)
            for col in ['x', 'y', 'w', 'h']:
                df[col] = df[col].round(2)
            
            df[self.output_cols].to_csv(os.path.join(self.output_dir, os.path.basename(fpath)), header=False, index=False)
        
        behav_files = glob.glob(os.path.join(self.input_dir, "behavior_*.txt"))
        for bf in behav_files:
            shutil.copy(bf, os.path.join(self.output_dir, os.path.basename(bf)))
        
        print(f"[CLEANER] Finished. Results in {self.output_dir}")