#!/bin/bash

current_directory = $(pwd)

# Start CUDA docker
docker run --rm \
    --gpus all \
    -v $current_directory:/app \
    nvidia/cuda:12.2.0-devel-ubuntu22.04 \
    /bin/bash -c "cd /app && nvcc -o output main.cu -run"