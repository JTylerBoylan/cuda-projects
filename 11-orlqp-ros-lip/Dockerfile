FROM ros:humble

RUN apt-get update -y && \
    apt-get install -y g++ cmake git gdb && \
    apt-get install -y libeigen3-dev

RUN git clone https://github.com/osqp/osqp
RUN cd osqp && mkdir build
RUN cd osqp/build && cmake -G "Unix Makefiles" ..
RUN cd osqp/build && cmake --build . --target install

ADD https://api.github.com/repos/JTylerBoylan/orlqp/git/refs/heads/main version.json
RUN git clone https://github.com/JTylerBoylan/orlqp
RUN cd orlqp && mkdir build
RUN cd orlqp/build && cmake ..
RUN cd orlqp/build && cmake --build . --target install

RUN ldconfig

WORKDIR /ros2_ws/

ARG USER_UID=1000
ARG USER_NAME=user
RUN useradd -m -u $USER_UID $USER_NAME && \
    chown -R $USER_NAME:$USER_NAME /ros2_ws/
USER $USER_NAME