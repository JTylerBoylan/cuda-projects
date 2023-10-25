#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

PROJECT_NAME="orlqp-ros-di2d"

docker build -t ${PROJECT_NAME} "${SCRIPT_DIR}"

docker run -it --rm \
    --net host \
    --privileged \
    -v $SCRIPT_DIR:/ros2_ws/src/orlqp_di2d/ \
    -v /dev/shm:/dev/shm \
    ${PROJECT_NAME} \
    bash