#!/usr/bin/env bash

# You may need to modify the following paths before compiling
CUDA_HOME=/home/wei/cuda-10.2 \
CUDNN_INCLUDE_DIR=/home/wei/cuda-10.2 \
CUDNN_LIB_DIR=/home/wei/cuda-10.2 \
python setup.py develop
