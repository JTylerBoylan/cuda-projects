FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update -y && \
    apt-get install -y cmake git gdb && \
    apt-get install -y libeigen3-dev

RUN git clone https://github.com/osqp/osqp
RUN cd osqp && mkdir build
# RUN cd osqp/build && cmake -G "Unix Makefiles" .. -DOSQP_ALGEBRA_BACKEND=cuda
RUN cd osqp/build && cmake -G "Unix Makefiles" ..
RUN cd osqp/build && cmake --build . --target install