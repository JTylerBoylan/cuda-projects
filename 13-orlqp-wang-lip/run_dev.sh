#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

PROJECT_NAME="orlqp-lip-wang"

docker build -t ${PROJECT_NAME} "${SCRIPT_DIR}"

docker run -it --rm \
    --net host \
    --privileged \
    -v $SCRIPT_DIR:/app \
    -v /dev/shm:/dev/shm \
    ${PROJECT_NAME} \
    bash
