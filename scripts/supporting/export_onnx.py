import torch
import sys

sys.path.append("/root/vision_pipeline/Depth-Anything-3/src")
from depth_anything_3.model.dinov2.dinov2 import DinoV2
from depth_anything_3.model.dpt import DPT
from depth_anything_3.model.da3 import DepthAnything3Net

def export_da3_to_onnx():
    print("Loading PyTorch model into GPU memory...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Constructing ViT-Large components...")
    net = DinoV2(name="vitl", out_layers=[4, 11, 17, 23], alt_start=-1, qknorm_start=-1, rope_start=-1, cat_token=False)
    head = DPT(dim_in=1024, output_dim=1, features=256, out_channels=[256, 512, 1024, 1024])
    
    model = DepthAnything3Net(net=net, head=head)
    
    print("Loading metric binary weights...")
    bin_path = "/root/vision_pipeline/DA3_Metric_Model/pytorch_model.bin"
    state_dict = torch.load(bin_path, map_location="cpu")
    
    print("Stripping legacy prefixes from matrix keys...")
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("model.", "") if key.startswith("model.") else key
        cleaned_state_dict[new_key] = value
        
    model.load_state_dict(cleaned_state_dict, strict=False)
    model = model.to(device).eval()

    print("Creating 5D dummy input tensor for sequence processing...")
    dummy_input = torch.randn(1, 1, 3, 518, 518, device=device)
    output_path = "/root/vision_pipeline/DA3_Metric_Model/da3_metric_518.onnx"

    print("Tracing mathematical operations to " + output_path)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['input_sequence'],
        output_names=['depth_map']
    )
    print("ONNX Export Complete.")

if __name__ == "__main__":
    export_da3_to_onnx()