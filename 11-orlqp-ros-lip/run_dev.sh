#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

PROJECT_NAME="orlqp-ros-lip"

docker build -t ${PROJECT_NAME} "${SCRIPT_DIR}"

docker run -it --rm \
    --net host \
    --privileged \
    -v $SCRIPT_DIR:/ros2_ws/src/orlqp_lip/ \
    -w /ros2_ws/src/ \
    ${PROJECT_NAME} \
    bash