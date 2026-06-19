import json
import numpy as np
import os
import sys

def generate_pose_graph(video_name):
    metadata_path = f"/root/vision_pipeline/uploads/{video_name}_metadata.json"
    status_path = f"/root/vision_pipeline/outputs/{video_name}/status.json"
    with open(metadata_path) as f:
        meta = json.load(f)

    poses_json = os.path.join(meta["paths"]["poses"], "camera_poses.json")
    viz_dir    = meta["paths"]["visualizations"]

    with open(poses_json) as f:
        poses = json.load(f)

    trajectory = []
    total_poses = len(poses.keys())
    
    for idx, key in enumerate(sorted(poses.keys())):
        matrix = np.array(poses[key])
        R = matrix[:3, :3]

        tx, ty, tz = matrix[0,3], matrix[1,3], matrix[2,3]

        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        if sy > 1e-6:
            roll  = np.degrees(np.arctan2(R[2,1], R[2,2]))
            pitch = np.degrees(np.arctan2(-R[2,0], sy))
            yaw   = np.degrees(np.arctan2(R[1,0], R[0,0]))
        else:
            roll  = np.degrees(np.arctan2(-R[1,2], R[1,1]))
            pitch = np.degrees(np.arctan2(-R[2,0], sy))
            yaw   = 0.0

        trajectory.append({
            "frame": key,
            "x": tx, "y": ty, "z": tz,
            "roll": roll, "pitch": pitch, "yaw": yaw
        })
        
        if idx % 50 == 0:
            try:
                with open(status_path, 'r') as f:
                    s_data = json.load(f)
                s_data['frame'] = idx
                s_data['total_frames'] = total_poses
                with open(status_path, 'w') as f:
                    json.dump(s_data, f)
            except Exception:
                pass

    out_path = os.path.join(viz_dir, "pose_graph.json")
    with open(out_path, "w") as f:
        json.dump(trajectory, f, indent=2)

    print(f"Pose graph saved to {out_path} ({len(trajectory)} poses)")
    return out_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_pose_graph.py <video_name>")
        sys.exit(1)
    generate_pose_graph(sys.argv[1])