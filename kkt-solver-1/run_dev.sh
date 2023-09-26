#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Start CUDA docker
docker run -it --rm \
    --gpus all \
    -v $SCRIPT_DIR:/app \
    -w /app \
    kkt-solver-1 \
    bash