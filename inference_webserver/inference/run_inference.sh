#!/usr/bin/env bash
cd "$(dirname "$0")"
nvidia-smi
python download_model_files.py
python generate_onnx.py
python trt_inference.py
sleep 99999999999