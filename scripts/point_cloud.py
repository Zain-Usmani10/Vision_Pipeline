import open3d as o3d
import numpy as np
import json
import cv2
import os
import glob
import sys

def run_point_cloud(video_name):
    metadata_path = f"/root/vision_pipeline/uploads/{video_name}_metadata.json"
    status_path = f"/root/vision_pipeline/outputs/{video_name}/status.json"
    with open(metadata_path) as f:
        meta = json.load(f)

    fx     = meta["intrinsics"]["fx"]
    fy     = meta["intrinsics"]["fy"]
    cx     = meta["intrinsics"]["cx"]
    cy     = meta["intrinsics"]["cy"]
    width  = meta["width"]
    height = meta["height"]

    frames_dir      = meta["paths"]["frames"]
    depth_dir       = meta["paths"]["depth"]
    pointclouds_dir = meta["paths"]["pointclouds"]
    poses_json      = os.path.join(meta["paths"]["poses"], "camera_poses.json")

    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

    with open(poses_json, "r") as f:
        poses = json.load(f)

    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    depth_files = sorted(glob.glob(os.path.join(depth_dir,  "*.npy")))

    if not frame_files:
        print("No frames found.")
        sys.exit(1)

    print(f"Generating point clouds for {video_name} ({len(frame_files)} frames)...")

    for idx, (frame_path, depth_path) in enumerate(zip(frame_files, depth_files)):
        frame_key = f"frame_{idx:04d}"
        if frame_key not in poses:
            continue

        color_rgb     = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
        raw_depth     = np.load(depth_path)
        depth_resized = cv2.resize(raw_depth, (width, height), interpolation=cv2.INTER_LINEAR)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_rgb),
            o3d.geometry.Image(depth_resized.astype(np.float32)),
            depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        pcd.transform(np.array(poses[frame_key]))

        out_path = os.path.join(pointclouds_dir, f"frame_{idx:04d}.ply")
        o3d.io.write_point_cloud(out_path, pcd.voxel_down_sample(0.02))

        if idx % 10 == 0:
            print(f"  Saved frame {idx}")
            try:
                with open(status_path, 'r') as f:
                    s_data = json.load(f)
                s_data['frame'] = idx
                with open(status_path, 'w') as f:
                    json.dump(s_data, f)
            except Exception:
                pass

    print(f"Point cloud generation complete. PLY files in {pointclouds_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python point_cloud.py <video_name>")
        sys.exit(1)
    run_point_cloud(sys.argv[1])