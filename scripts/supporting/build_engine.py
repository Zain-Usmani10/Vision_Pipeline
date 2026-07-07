import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine():
    onnx_file_path = "/root/vision_pipeline/DA3_Metric_Model/da3_metric_518.onnx"
    engine_file_path = "/root/vision_pipeline/DA3_Metric_Model/da3_metric_518_fp16.engine"

    print("Initializing TensorRT compiler...")
    builder = trt.Builder(TRT_LOGGER)

    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print("Parsing ONNX blueprint...")
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            return

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 * 1024 * 1024 * 1024)

    # TRT 11.1: FP16 is set directly as a property on config, not a BuilderFlag enum
    config.set_flag(trt.BuilderFlag(1))

    print("Building engine (this will take several minutes)...")
    engine_bytes = builder.build_serialized_network(network, config)

    if engine_bytes is None:
        print("Compilation FAILED.")
        return

    with open(engine_file_path, "wb") as f:
        f.write(engine_bytes)

    print(f"Engine saved to {engine_file_path}")

if __name__ == "__main__":
    build_engine()