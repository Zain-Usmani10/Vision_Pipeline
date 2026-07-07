import cv2
import numpy as np
import torch
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class DA3DepthWorker:
    def __init__(self, engine_path, input_queue, output_queue, focal_length):
        self.engine_path = engine_path
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.focal_length = focal_length

    def run(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        runtime = trt.Runtime(TRT_LOGGER)
        
        with open(self.engine_path, "rb") as f:
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

        while True:
            item = self.input_queue.get()
            if item is None:
                break
                
            frame, frame_idx = item
            orig_h, orig_w = frame.shape[:2]
            
            PATCH_SIZE = 14
            scale = 504.0 / max(orig_h, orig_w)
            new_h = round(orig_h * scale / PATCH_SIZE) * PATCH_SIZE
            new_w = round(orig_w * scale / PATCH_SIZE) * PATCH_SIZE
            img_resized = cv2.resize(frame, (new_w, new_h))

            # Pad to engine input size (518x518)
            pad_h = 518 - new_h
            pad_w = 518 - new_w
            img_padded = cv2.copyMakeBorder(img_resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                        
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
            
            metric_depth = depth_restored
            self.output_queue.put((frame, metric_depth, frame_idx))