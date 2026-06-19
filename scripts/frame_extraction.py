import cv2
import os
import json
import sys

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

    count = 0
    print(f"Extracting frames from {video_path}...")
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        cv2.imwrite(f"{frames_dir}/frame_{count:04d}.png", frame)
        
        # --- ADD THIS BLOCK TO UPDATE PROGRESS ---
        if count % 10 == 0: 
            try:
                # Load the current status
                with open(status_path, 'r') as f:
                    s_data = json.load(f)
                # Update only the frame count
                s_data['frame'] = count
                # Write back
                with open(status_path, 'w') as f:
                    json.dump(s_data, f)
            except Exception as e:
                print(f"Status update error: {e}")
        # ------------------------------------------
        
        count += 1
        if count >= 9999999:
            break

    video.release()
    metadata["frame_count"] = count

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done. Extracted {count} frames to {frames_dir}")
    print(f"Metadata saved to {metadata_path}")
    print(f"Intrinsics: fx=fy={fx}, cx={cx}, cy={cy}")

    return metadata

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python frame_extraction.py <video_path>")
        sys.exit(1)
    extract_frames(sys.argv[1])