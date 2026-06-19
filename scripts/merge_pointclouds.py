import open3d as o3d
import glob
import os
import sys
import json

def merge_pointclouds(video_name):
    metadata_path = f"/root/vision_pipeline/uploads/{video_name}_metadata.json"
    status_path = f"/root/vision_pipeline/outputs/{video_name}/status.json"
    with open(metadata_path) as f:
        meta = json.load(f)

    pointclouds_dir = meta["paths"]["pointclouds"]
    ply_files = sorted(glob.glob(os.path.join(pointclouds_dir, "*.ply")))

    if not ply_files:
        print("No PLY files found.")
        sys.exit(1)

    print(f"Merging {len(ply_files)} point clouds...")

    combined = o3d.geometry.PointCloud()
    for idx, ply_path in enumerate(ply_files):
        pcd = o3d.io.read_point_cloud(ply_path)
        combined += pcd
        
        if idx % 50 == 0:
            print(f"  Merged {idx}/{len(ply_files)}")
            try:
                with open(status_path, 'r') as f:
                    s_data = json.load(f)
                s_data['frame'] = idx
                s_data['total_frames'] = len(ply_files)
                with open(status_path, 'w') as f:
                    json.dump(s_data, f)
            except Exception:
                pass

    combined = combined.voxel_down_sample(0.05)

    out_path = os.path.join(pointclouds_dir, "merged.ply")
    o3d.io.write_point_cloud(out_path, combined)
    print(f"Merged point cloud saved to {out_path}")
    print(f"Total points: {len(combined.points)}")

    return out_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_pointclouds.py <video_name>")
        sys.exit(1)
    merge_pointclouds(sys.argv[1])