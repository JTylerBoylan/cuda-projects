#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

PROJECT_NAME="7-osqp-benchmark"

docker build -t ${PROJECT_NAME} "${SCRIPT_DIR}"

# Start CUDA docker
docker run -it --rm \
    --gpus all \
    -v $SCRIPT_DIR:/app \
    -w /app \
    ${PROJECT_NAME} \
    bash