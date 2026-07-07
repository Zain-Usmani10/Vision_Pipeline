import cv2
import numpy as np
import torch
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class DA3TensorRTInference:
    def __init__(self, engine_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self.outputs = {}
        
        # Dynamically allocate memory for every tensor the engine expects
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            shape = tuple(self.engine.get_tensor_shape(name))
            
            tensor = torch.empty(shape, dtype=torch.float32, device=self.device)
            self.context.set_tensor_address(name, tensor.data_ptr())
            
            if is_input:
                self.input_tensor = tensor
            else:
                self.outputs[name] = tensor

    def preprocess(self, image_path):
        img = cv2.imread(image_path)
        orig_h, orig_w = img.shape[:2]
        
        scale = 504.0 / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        
        pad_h = 518 - new_h
        pad_w = 518 - new_w
        img_padded = cv2.copyMakeBorder(img_resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_rgb - mean) / std
        
        img_chw = img_norm.transpose(2, 0, 1)
        
        # Reshape to match the exact dimensions TRT demands
        img_tensor = torch.from_numpy(img_chw).view(self.input_tensor.shape)
        self.input_tensor.copy_(img_tensor)
        
        return orig_h, orig_w, new_h, new_w

    def infer(self):
        self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        # Pull only the depth map output we care about
        return self.outputs["depth_map"]

    def postprocess(self, raw_output, orig_h, orig_w, new_h, new_w, focal_length):
        depth = raw_output.squeeze().cpu().numpy()
        
        depth_cropped = depth[:new_h, :new_w]
        
        depth_restored = cv2.resize(depth_cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        
        metric_depth = depth_restored * focal_length
        
        return metric_depth

def main():
    engine_path = "/root/vision_pipeline/DA3_Metric_Model/da3_metric_518_fp16.engine"
    image_path = "/root/vision_pipeline/scripts/frame_0224.png"
    
    width = 1920
    fov_degrees = 80
    focal_length = (width / 2.0) / np.tan(np.radians(fov_degrees / 2.0))
    
    print("Loading optimized TensorRT Engine")
    inferencer = DA3TensorRTInference(engine_path)
    
    print("Preprocessing image with ImageNet normalization and padding")
    orig_h, orig_w, new_h, new_w = inferencer.preprocess(image_path)
    
    print("Executing zero copy VRAM inference")
    raw_depth = inferencer.infer()
    
    print("Postprocessing output to metric scale")
    metric_depth = inferencer.postprocess(raw_depth, orig_h, orig_w, new_h, new_w, focal_length)
    
    print(f"Metric depth min {metric_depth.min():.2f}m max {metric_depth.max():.2f}m mean {metric_depth.mean():.2f}m")
    
    normalized = ((metric_depth - metric_depth.min()) / (metric_depth.max() - metric_depth.min()) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_MAGMA)
    cv2.imwrite("/root/vision_pipeline/outputs/trt_depth_preview.png", colored)
    print("Saved preview to outputs folder")

if __name__ == "__main__":
    main()