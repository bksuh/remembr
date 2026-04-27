"""Bake `trajectory_gt_object_v2.py` per-row debug output into a static site
under `web_viewer/site/` so it can be hosted on GitHub Pages.

For each CSV row, writes:
  - site/meta/<idx>.json           per-row data for the Plotly viewer
  - site/frames/<idx>/<i>.jpg      cam0 RGB frame (i=0..5), JPEG q=<quality>

Plus once at the end:
  - site/index.json                small list of rows for the sidebar picker

Reuses helpers from trajectory_gt_object_v2.py so the bake stays in lockstep
with the live debug script.
"""

import argparse
import json
import os
import pickle as pkl
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from PIL import Image

# Make the parent folder importable so `from trajectory_gt_object_v2 import ...` works
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

from trajectory_gt_object_v2 import (  # noqa: E402
    CAPTIONS_ROOT,
    CAPTION_FNAME,
    CSV_PATH,
    NAVQA_FNAME,
    NAVQA_ROOT,
    circular_mean_deg,
    load_frame_poses,
    parse_hms_to_epoch,
    parse_text_answer_position,
    remap_pkl_path,
    snap_to_window,
)


def bake_row(idx, row, out_dir, quality, max_side, recording_tz):
    seq_id = int(row["Seq ID"])
    question = str(row["Question"])
    sequence_uuid = str(row["UUID"])
    hms_time = str(row["Timestamp with answer"]).strip()

    entries_path = f"{CAPTIONS_ROOT}/{seq_id}/captions/{CAPTION_FNAME}"
    navqa_path = f"{NAVQA_ROOT}/{seq_id}/{NAVQA_FNAME}"

    with open(entries_path, "r") as f:
        entries = json.load(f)
    with open(navqa_path, "r") as f:
        navqa = json.load(f)

    sequence = next((s for s in navqa["data"] if s["id"] == sequence_uuid), None)
    if sequence is None:
        return {
            "idx": idx,
            "seq_id": seq_id,
            "uuid": sequence_uuid,
            "question": question,
            "status": "missing_navqa",
        }

    gt_epoch = parse_hms_to_epoch(hms_time, sequence["start_time"], recording_tz=recording_tz)
    caption_idx, in_window, offset = snap_to_window(entries, sequence, gt_epoch)
    if caption_idx is None:
        return {
            "idx": idx,
            "seq_id": seq_id,
            "uuid": sequence_uuid,
            "question": question,
            "status": "no_window_captions",
        }

    cap = entries[caption_idx]
    used_files = cap["used_files"]
    frame_poses = load_frame_poses(used_files)

    if frame_poses:
        gt_caption_yaw = circular_mean_deg([yaw for _, _, yaw in frame_poses])
        fx0, fy0, _ = frame_poses[0]
        fxN, fyN, _ = frame_poses[-1]
        motion_heading = float(np.degrees(np.arctan2(fyN - fy0, fxN - fx0)))
    else:
        gt_caption_yaw = None
        motion_heading = None

    gt_truth = parse_text_answer_position(row["Text answer"])

    frames_dir = os.path.join(out_dir, "frames", str(idx))
    os.makedirs(frames_dir, exist_ok=True)
    frames_rel_paths = []
    for i, fp in enumerate(used_files):
        rp = remap_pkl_path(fp)
        if not os.path.exists(rp):
            frames_rel_paths.append(None)
            continue
        with open(rp, "rb") as f:
            d = pkl.load(f)
        rgb = d["cam0"][..., ::-1]
        img = Image.fromarray(rgb)
        if max_side:
            img.thumbnail((max_side, max_side), Image.LANCZOS)
        out_jpg = os.path.join(frames_dir, f"{i}.jpg")
        img.save(out_jpg, "JPEG", quality=quality, optimize=True, progressive=True)
        frames_rel_paths.append(f"frames/{idx}/{i}.jpg")

    status = "in_window" if in_window else "snapped"

    meta = {
        "idx": idx,
        "seq_id": seq_id,
        "uuid": sequence_uuid,
        "question": question,
        "csv_hms": hms_time,
        "gt_epoch": float(gt_epoch),
        "gt_iso": datetime.fromtimestamp(gt_epoch, tz=timezone.utc).isoformat(),
        "window_epoch": [float(sequence["start_time"]), float(sequence["end_time"])],
        "window_iso": [
            datetime.fromtimestamp(sequence["start_time"], tz=timezone.utc).isoformat(),
            datetime.fromtimestamp(sequence["end_time"], tz=timezone.utc).isoformat(),
        ],
        "in_window": bool(in_window),
        "status": status,
        "snap_offset_seconds": float(gt_epoch - sequence["start_time"]) if not in_window else 0.0,
        "caption_idx": int(caption_idx),
        "caption_time": float(cap["time"]),
        "caption_position_xy": [float(cap["position"][0]), float(cap["position"][1])],
        "trajectory_xy": [[float(e["position"][0]), float(e["position"][1])] for e in entries],
        "in_window_xy": [
            [float(e["position"][0]), float(e["position"][1])]
            for e in entries
            if sequence["start_time"] <= e["time"] <= sequence["end_time"]
        ],
        "gt_caption": {
            "x": float(cap["position"][0]),
            "y": float(cap["position"][1]),
            "yaw_deg": gt_caption_yaw,
        },
        "frame_poses": [[float(x), float(y), float(yaw)] for (x, y, yaw) in frame_poses],
        "gt_truth_xyz": [list(map(float, p)) for p in gt_truth],
        "yaw_check": {
            "theta0_deg": float(cap["theta"][0]),
            "true_yaw_deg": gt_caption_yaw,
            "motion_heading_deg": motion_heading,
        },
        "frames": frames_rel_paths,
        "fov_deg": 70.0,
        "fov_radius": 7.0,
        "frame_fov_radius": 7.0,
    }

    meta_dir = os.path.join(out_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, f"{idx}.json"), "w") as f:
        json.dump(meta, f)

    return {
        "idx": idx,
        "seq_id": seq_id,
        "uuid": sequence_uuid,
        "question": question,
        "status": status,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row", type=int, default=None, help="Bake only this CSV row index")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality (default 85)")
    parser.add_argument("--max-side", type=int, default=None,
                        help="Downscale longest side to this many pixels (default: native)")
    parser.add_argument("--out", default=None,
                        help="Output dir (default: web_viewer/site)")
    parser.add_argument("--tz", default="US/Pacific", help="Recording timezone for HMS")
    parser.add_argument("--clean", action="store_true",
                        help="Wipe site/frames and site/meta before baking")
    args = parser.parse_args()

    out_dir = args.out or os.path.join(HERE, "site")
    os.makedirs(out_dir, exist_ok=True)

    if args.clean:
        for sub in ("frames", "meta"):
            d = os.path.join(out_dir, sub)
            if os.path.exists(d):
                shutil.rmtree(d)

    df = pd.read_csv(CSV_PATH)
    rows_iter = df.iterrows() if args.row is None else [(args.row, df.iloc[args.row])]

    index_entries = []
    for idx, row in rows_iter:
        try:
            entry = bake_row(int(idx), row, out_dir,
                             quality=args.quality, max_side=args.max_side,
                             recording_tz=args.tz)
        except Exception as e:
            print(f"[{idx}] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            entry = {
                "idx": int(idx), "seq_id": int(row["Seq ID"]),
                "uuid": str(row["UUID"]), "question": str(row["Question"]),
                "status": f"error:{type(e).__name__}",
            }
        print(f"[{entry['idx']}] {entry['status']:<20} seq {entry['seq_id']} "
              f"{entry['uuid']}: {entry['question'][:60]}")
        index_entries.append(entry)

    if args.row is None:
        with open(os.path.join(out_dir, "index.json"), "w") as f:
            json.dump(index_entries, f, indent=2)
        print(f"\nWrote index.json with {len(index_entries)} rows -> {out_dir}/index.json")


if __name__ == "__main__":
    main()
