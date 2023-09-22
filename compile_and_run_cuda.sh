#!/bin/bash

CURRENT_DIR=$(pwd)

# Start CUDA docker
docker run --rm \
    --gpus all \
    -v $CURRENT_DIR:/app \
    -w /app \
    nvidia/cuda:12.2.0-devel-ubuntu22.04 \
    /bin/bash -c "make run"
