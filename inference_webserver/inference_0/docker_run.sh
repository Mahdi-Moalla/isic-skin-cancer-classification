#!/usr/bin/env bash
cd "$(dirname "$0")"
docker run -it --runtime=nvidia --gpus all\
     --ipc=host --ulimit memlock=-1\
     --ulimit stack=67108864\
     -v $(pwd):/workspace\
     nvcr_pytorch_tensorrt_mod:latest bash 

#-v ../project_data_prepare/split_dataset:/dataset\