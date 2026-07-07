import cv2
import numpy as np
import collections

class SlidingWindowSLAM:
    def __init__(self, input_queue, output_queue, focal_length, width, height):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.focal_length = focal_length
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.camera_matrix = np.array([[focal_length, 0, self.cx], [0, focal_length, self.cy], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1))
        self.window = collections.deque(maxlen=10)
        self.feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=10, blockSize=3)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.current_R = np.eye(3)
        self.current_t = np.zeros((3, 1))

    def run(self):
        prev_gray = None
        prev_pts = None
        prev_depth = None
        
        while True:
            item = self.input_queue.get()
            if item is None:
                break
                
            frame, depth_map, frame_idx = item
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            extrinsic_4x4 = np.eye(4)
            
            if prev_gray is None:
                prev_gray = gray
                prev_depth = depth_map
                prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)
                self.window.append((frame, depth_map, extrinsic_4x4, frame_idx))
                continue
                
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **self.lk_params)
            good_prev = prev_pts[status == 1]
            good_curr = curr_pts[status == 1]
            
            points_3d = []
            valid_curr_pts = []
            
            for pt_prev, pt_curr in zip(good_prev, good_curr):
                u, v = int(pt_prev[0]), int(pt_prev[1])
                if v >= depth_map.shape[0] or u >= depth_map.shape[1]:
                    continue
                z = prev_depth[v, u]
                if z <= 0.1:
                    continue
                x = (u - self.cx) * z / self.focal_length
                y = (v - self.cy) * z / self.focal_length
                points_3d.append([x, y, z])
                valid_curr_pts.append(pt_curr)
                
            points_3d = np.array(points_3d, dtype=np.float32)
            valid_curr_pts = np.array(valid_curr_pts, dtype=np.float32)
            
            if len(points_3d) >= 6:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, valid_curr_pts, self.camera_matrix, self.dist_coeffs, reprojectionError=1.5, confidence=0.995)
                if success:
                    R, _ = cv2.Rodrigues(rvec)
                    R_inv = R.T
                    t_inv = -R_inv.dot(tvec)
                    self.current_t = self.current_t + self.current_R.dot(t_inv)
                    self.current_R = self.current_R.dot(R_inv)
                    
            extrinsic_4x4[:3, :3] = self.current_R
            extrinsic_4x4[:3, 3] = self.current_t.flatten()
            
            if len(self.window) == 10:
                evicted_frame, evicted_depth, evicted_pose, evicted_idx = self.window[0]
                self.output_queue.put((evicted_frame, evicted_depth, evicted_pose, evicted_idx))
                
            self.window.append((frame, depth_map, extrinsic_4x4, frame_idx))
            prev_gray = gray
            prev_depth = depth_map
            prev_pts = cv2.goodFeaturesToTrack(gray, mask=None, **self.feature_params)