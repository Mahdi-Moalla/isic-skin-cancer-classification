import fire
import os

# This sample uses an ONNX ResNet50 Model to create a TensorRT Inference Engine
import random
import sys

import numpy as np

import tensorrt as trt
from PIL import Image

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

import  time

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine_onnx(model_file):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)
    config = builder.create_builder_config()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, common.GiB(1))
    # Load the Onnx model and parse it in order to populate the TensorRT network.
    with open(model_file, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    engine_bytes = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(TRT_LOGGER)
    return runtime.deserialize_cuda_engine(engine_bytes)





def onnx_to_trt(onnx_model_path='./model.onnx'):
    engine = build_engine_onnx(onnx_model_path)

    ts=[]
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        
    context = engine.create_execution_context()

    for  _ in range(1000):
        t0=time.time()
        
        np.copyto(inputs[0].host, np.zeros(( 3,224,224) ).ravel())
        np.copyto(inputs[1].host, np.zeros(32).ravel())

        trt_outputs = common.do_inference(
            context,
            engine=engine,
            bindings=bindings,
            inputs=inputs,
            outputs=outputs,
            stream=stream,
        )
        # We use the highest probability as our prediction. Its index corresponds to the predicted label.
        pred = np.argmax(trt_outputs[0])
        print(pred)

        t1=time.time()

        ts.append(t1-t0)
    
    print(np.array(ts).mean())
    common.free_buffers(inputs, outputs, stream)
    

    from IPython import embed; embed(colors='Linux') 

if __name__=='__main__':
    fire.Fire(onnx_to_trt)