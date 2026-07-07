import cv2
import os
import json
import sys
from stream_gatekeeper import MotionGatekeeper

def extract_frames(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir   = f"/root/vision_pipeline/outputs/{video_name}/frames"
    metadata_path = f"/root/vision_pipeline/uploads/{video_name}_metadata.json"
    status_path = f"/root/vision_pipeline/outputs/{video_name}/status.json"
    
    os.makedirs(frames_dir, exist_ok=True)
    for folder in ["frames", "depth", "poses", "pointclouds", "visualizations"]:
        os.makedirs(f"/root/vision_pipeline/outputs/{video_name}/{folder}", exist_ok=True)

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"ERROR: Could not open video {video_path}")
        sys.exit(1)

    width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = video.get(cv2.CAP_PROP_FPS)

    fx = fy = float(max(width, height))
    cx = width  / 2.0
    cy = height / 2.0

    metadata = {
        "video_name": video_name,
        "video_path": video_path,
        "width":  width,
        "height": height,
        "fps":    fps,
        "intrinsics": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy
        },
        "paths": {
            "frames":         f"/root/vision_pipeline/outputs/{video_name}/frames",
            "depth":          f"/root/vision_pipeline/outputs/{video_name}/depth",
            "poses":          f"/root/vision_pipeline/outputs/{video_name}/poses",
            "pointclouds":    f"/root/vision_pipeline/outputs/{video_name}/pointclouds",
            "visualizations": f"/root/vision_pipeline/outputs/{video_name}/visualizations"
        }
    }

    # Initialize the gatekeeper to prune stationary frames
    gatekeeper = MotionGatekeeper(threshold=25, min_change_ratio=0.01)
    
    count_seen = 0
    count_saved = 0
    
    print(f"Extracting frames from {video_path}...")
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
            
        # Ask the gatekeeper if this frame has enough motion to matter
        if not gatekeeper.should_keep_frame(frame):
            count_seen += 1
            continue

        # We only save sequentially based on what was KEPT
        cv2.imwrite(f"{frames_dir}/frame_{count_saved:04d}.png", frame)
        
        # Asynchronous progress update hook
        if count_saved % 10 == 0: 
            try:
                with open(status_path, 'r') as f:
                    s_data = json.load(f)
                s_data['frame'] = count_saved
                with open(status_path, 'w') as f:
                    json.dump(s_data, f)
            except Exception as e:
                pass
        
        count_saved += 1
        count_seen += 1
        
        # Failsafe limit
        if count_seen >= 9999999:
            break

    video.release()
    
    # Print the motion elimination statistics
    gatekeeper.print_summary()
    
    # Save metadata with the exact number of frames that survived the filter
    metadata["frame_count"] = count_saved

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done. Extracted {count_saved} frames to {frames_dir}")
    print(f"Metadata saved to {metadata_path}")
    print(f"Intrinsics: fx=fy={fx}, cx={cx}, cy={cy}")

    return metadata

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python frame_extraction.py <video_path>")
        sys.exit(1)
    extract_frames(sys.argv[1])