import sys
import os

# Add scripts to path
sys.path.append("/root/vision_pipeline/scripts")

from frame_extraction    import extract_frames
from vid_depth           import run_depth
from pose_estimation     import run_pose_estimation
from point_cloud         import run_point_cloud
from merge_pointclouds   import merge_pointclouds
from generate_pose_graph import generate_pose_graph

def run_pipeline(video_path):
    print("\n" + "="*50)
    print("VISION PIPELINE STARTING")
    print("="*50)

    # Step 1 — Frame extraction + metadata + folder creation
    print("\n[1/6] Extracting frames...")
    meta = extract_frames(video_path)
    video_name = meta["video_name"]

    # Step 2 — Depth estimation
    print("\n[2/6] Running depth estimation...")
    run_depth(video_name)

    # Step 3 — Pose estimation
    print("\n[3/6] Running pose estimation...")
    run_pose_estimation(video_name)

    # Step 4 — Point cloud generation
    print("\n[4/6] Generating point clouds...")
    run_point_cloud(video_name)

    # Step 5 — Merge point clouds
    print("\n[5/6] Merging point clouds...")
    merge_pointclouds(video_name)

    # Step 6 — Generate pose graph data
    print("\n[6/6] Generating pose graph...")
    generate_pose_graph(video_name)

    print("\n" + "="*50)
    print("PIPELINE COMPLETE")
    print(f"Outputs in: /root/vision_pipeline/outputs/{video_name}/")
    print("="*50)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path>")
        sys.exit(1)
    run_pipeline(sys.argv[1])