import os
import glob
import cv2
import numpy as np
import torch
import tensorrt as trt
import open3d as o3d

focal_length = 1386.66
engine_path = "/root/vision_pipeline/DA3_Metric_Model/da3_metric_518_fp16.engine"
output_dir = "/root/vision_pipeline/outputs/supporting"

os.makedirs(output_dir, exist_ok=True)

upload_dir = "/root/vision_pipeline/uploads"
video_files = glob.glob(os.path.join(upload_dir, "*.mp4"))
if not video_files:
    print("CRITICAL ERROR: No MP4 file found in uploads directory.")
    exit()

video_path = video_files[0]
print(f"Targeting video: {video_path}")

print("Loading TensorRT Engine...")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
runtime = trt.Runtime(TRT_LOGGER)

with open(engine_path, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()
outputs = {}

for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    is_input = engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
    shape = tuple(engine.get_tensor_shape(name))
    tensor = torch.empty(shape, dtype=torch.float32, device=device)
    context.set_tensor_address(name, tensor.data_ptr())
    if is_input:
        input_tensor = tensor
    else:
        outputs[name] = tensor

target_frames = [50, 150, 250, 400]
cap = cv2.VideoCapture(video_path)

for frame_idx in target_frames:
    print(f"\nProcessing Frame {frame_idx}...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame {frame_idx}. Video might be too short.")
        continue

    orig_h, orig_w = frame.shape[:2]

    scale = 504.0 / max(orig_h, orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
    img_resized = cv2.resize(frame, (new_w, new_h))

    pad_h = 518 - new_h
    pad_w = 518 - new_w
    img_padded = cv2.copyMakeBorder(img_resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_rgb - mean) / std

    img_chw = img_norm.transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img_chw).view(input_tensor.shape).to(device)
    input_tensor.copy_(img_tensor)

    context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    raw_output = outputs["depth_map"].squeeze().cpu().numpy()
    depth_cropped = raw_output[:new_h, :new_w]
    depth_restored = cv2.resize(depth_cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    depth_normalized = cv2.normalize(depth_restored, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
    depth_png_path = os.path.join(output_dir, f"depth_frame_{frame_idx}.png")
    cv2.imwrite(depth_png_path, depth_colored)
    print(f"Saved {depth_png_path}")

    metric_depth = (depth_restored * focal_length) / 1000.0

    cx = orig_w / 2.0
    cy = orig_h / 2.0
    intrinsic = o3d.camera.PinholeCameraIntrinsic(orig_w, orig_h, focal_length, focal_length, cx, cy)

    color_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color_rgb),
        o3d.geometry.Image(metric_depth.astype(np.float32)),
        depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    ply_path = os.path.join(output_dir, f"diagnostic_frame_{frame_idx}.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"Saved {ply_path}")

cap.release()
print("\nDiagnostic extraction complete.")