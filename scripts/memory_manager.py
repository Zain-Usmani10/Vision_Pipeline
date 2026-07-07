import open3d as o3d
import numpy as np
import cv2
import os
import glob

class SubMapManager:
    def __init__(self, input_queue, focal_length, width, height, status_dict=None):
        self.input_queue = input_queue
        self.focal_length = focal_length
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.global_pcd = o3d.geometry.PointCloud()
        self.chunk_threshold = 100
        self.frames_processed = 0
        self.chunk_index = 0
        self.total_points = 0
        self.output_dir = "/root/vision_pipeline/outputs/active_sessions"
        os.makedirs(self.output_dir, exist_ok=True)
        self.intrinsic = None
        self.status_dict = status_dict
        self.voxel_size = 0.02

    def run(self):
        while True:
            item = self.input_queue.get()
            if item is None:
                print("[Memory Worker] Merging chunks for final output...")
                final_cloud = o3d.geometry.PointCloud()
                
                if not self.global_pcd.is_empty():
                    out_path = os.path.join(self.output_dir, "submap_chunk_final.ply")
                    o3d.io.write_point_cloud(out_path, self.global_pcd)
                
                chunk_files = glob.glob(os.path.join(self.output_dir, "submap_chunk_*.ply"))
                for file in chunk_files:
                    chunk = o3d.io.read_point_cloud(file)
                    final_cloud += chunk
                
                final_cloud = final_cloud.voxel_down_sample(self.voxel_size)
                out_path = os.path.join(self.output_dir, "final_submap.ply")
                o3d.io.write_point_cloud(out_path, final_cloud)
                
                if self.status_dict is not None:
                    self.status_dict["point_count"] = len(final_cloud.points)
                    
                print("[Memory Worker] Final submap exported successfully")
                break
                
            frame, depth_map, pose, frame_idx = item
            
            if self.intrinsic is None:
                h, w = frame.shape[:2]
                self.intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, self.focal_length, self.focal_length, self.cx, self.cy)
            
            color_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_rgb),
                o3d.geometry.Image(depth_map.astype(np.float32)),
                depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False
            )
            
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsic)
            pcd.transform(pose)
            self.global_pcd += pcd.voxel_down_sample(self.voxel_size)
            
            current_chunk_points = len(self.global_pcd.points)
            display_points = self.total_points + current_chunk_points
            
            if self.status_dict is not None:
                self.status_dict["frame_idx"] = frame_idx
                self.status_dict["point_count"] = display_points
            
            self.frames_processed += 1
            if self.frames_processed >= self.chunk_threshold:
                out_path = os.path.join(self.output_dir, f"submap_chunk_{self.chunk_index}.ply")
                o3d.io.write_point_cloud(out_path, self.global_pcd)
                self.total_points += current_chunk_points
                self.global_pcd = o3d.geometry.PointCloud()
                self.frames_processed = 0
                self.chunk_index += 1