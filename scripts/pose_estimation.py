import cv2
import numpy as np
import os
import glob
import json
import sys

def run_pose_estimation(video_name):
    metadata_path = f"/root/vision_pipeline/uploads/{video_name}_metadata.json"
    status_path = f"/root/vision_pipeline/outputs/{video_name}/status.json"
    with open(metadata_path) as f:
        meta = json.load(f)

    fx = meta["intrinsics"]["fx"]
    fy = meta["intrinsics"]["fy"]
    cx = meta["intrinsics"]["cx"]
    cy = meta["intrinsics"]["cy"]

    frames_dir  = meta["paths"]["frames"]
    depth_dir   = meta["paths"]["depth"]
    poses_dir   = meta["paths"]["poses"]
    width       = meta["width"]
    height      = meta["height"]

    output_json = os.path.join(poses_dir, "camera_poses.json")

    camera_matrix = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ], dtype=np.float32)

    dist_coeffs = np.zeros((4, 1))

    frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    depth_files = sorted(glob.glob(os.path.join(depth_dir,  "*.npy")))

    if not frame_files:
        print("No frames found.")
        sys.exit(1)
    if not depth_files:
        print("No depth maps found.")
        sys.exit(1)

    current_R = np.eye(3)
    current_t = np.zeros((3, 1))

    feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=10, blockSize=3)
    lk_params = dict(
        winSize=(15, 15), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    prev_gray  = None
    prev_pts   = None
    prev_depth = None
    pose_data  = {}

    print(f"Starting pose estimation for {video_name} ({len(frame_files)} frames)")
    print(f"Intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")

    for idx, (frame_path, depth_path) in enumerate(zip(frame_files, depth_files)):
        frame = cv2.imread(frame_path)
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        raw_depth = np.load(depth_path)
        depth_map = cv2.resize(raw_depth, (width, height), interpolation=cv2.INTER_LINEAR)

        extrinsic_4x4 = np.eye(4)

        if prev_gray is None:
            prev_gray  = gray
            prev_depth = depth_map
            prev_pts   = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)
            pose_data[f"frame_{idx:04d}"] = extrinsic_4x4.tolist()
            continue

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)

        good_prev = prev_pts[status == 1]
        good_curr = curr_pts[status == 1]

        points_3d      = []
        valid_curr_pts = []

        for pt_prev, pt_curr in zip(good_prev, good_curr):
            u, v = int(pt_prev[0]), int(pt_prev[1])
            z = prev_depth[v, u]
            if z <= 0.1:
                continue
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            points_3d.append([x, y, z])
            valid_curr_pts.append(pt_curr)

        points_3d      = np.array(points_3d,      dtype=np.float32)
        valid_curr_pts = np.array(valid_curr_pts,  dtype=np.float32)

        if len(points_3d) >= 6:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d, valid_curr_pts, camera_matrix, dist_coeffs,
                reprojectionError=1.5, confidence=0.995
            )
            if success:
                R, _   = cv2.Rodrigues(rvec)
                R_inv  = R.T
                t_inv  = -R_inv.dot(tvec)
                current_t = current_t + current_R.dot(t_inv)
                current_R = current_R.dot(R_inv)      

        extrinsic_4x4[:3, :3] = current_R
        extrinsic_4x4[:3,  3] = current_t.flatten()
        pose_data[f"frame_{idx:04d}"] = extrinsic_4x4.tolist()

        prev_gray  = gray
        prev_depth = depth_map
        prev_pts   = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)

        if idx % 10 == 0:
            print(f"  Frame {idx} | X={current_t[0][0]:.2f} Y={current_t[1][0]:.2f} Z={current_t[2][0]:.2f}")
            try:
                with open(status_path, 'r') as f:
                    s_data = json.load(f)
                s_data['frame'] = idx
                with open(status_path, 'w') as f:
                    json.dump(s_data, f)
            except Exception:
                pass

    with open(output_json, "w") as f:
        json.dump(pose_data, f, indent=4)

    print(f"Pose estimation complete. Saved to {output_json}")
    return output_json

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pose_estimation.py <video_name>")
        sys.exit(1)
    run_pose_estimation(sys.argv[1])