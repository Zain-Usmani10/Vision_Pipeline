import cv2
import torch
import numpy as np
import sys
import os
import glob
import json

sys.path.append("/root/vision_pipeline/Depth-Anything-3/src")
from depth_anything_3.api import DepthAnything3

def run_depth(video_name):
    metadata_path = f"/root/vision_pipeline/uploads/{video_name}_metadata.json"
    status_path = f"/root/vision_pipeline/outputs/{video_name}/status.json"
    with open(metadata_path) as f:
        meta = json.load(f)

    width      = meta["width"]
    height     = meta["height"]
    fx         = meta["intrinsics"]["fx"]
    frames_dir = meta["paths"]["frames"]
    depth_dir  = meta["paths"]["depth"]
    viz_dir    = meta["paths"]["visualizations"]

    scale_factor = fx / 300.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DepthAnything3.from_pretrained("/root/vision_pipeline/DA3_Metric_Model")
    model = model.to(device=device).eval()

    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    if not frame_files:
        print("No frames found.")
        sys.exit(1)

    print(f"Processing {len(frame_files)} frames with scale_factor={scale_factor:.4f}...")

    for idx, frame_path in enumerate(frame_files):
        prediction = model.inference([frame_path])
        depth_map = np.array(prediction.depth[0]) * scale_factor
        np.save(os.path.join(depth_dir, f"depth_{idx:04d}.npy"), depth_map)

        normalized = (depth_map / depth_map.max() * 255).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
        colored_resized = cv2.resize(colored, (width, height), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(viz_dir, f"depth_{idx:04d}.png"), colored_resized)

        if idx % 10 == 0:
            print(f"  Frame {idx}/{len(frame_files)}")
            try:
                with open(status_path, 'r') as f:
                    s_data = json.load(f)
                s_data['frame'] = idx
                with open(status_path, 'w') as f:
                    json.dump(s_data, f)
            except Exception:
                pass

    print(f"Depth estimation complete. NPYs in {depth_dir}, visualizations in {viz_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python vid_depth.py <video_name>")
        sys.exit(1)
    run_depth(sys.argv[1])