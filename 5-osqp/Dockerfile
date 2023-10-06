FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update -y

RUN apt-get install -y cmake git

RUN apt-get install -y libeigen3-dev

RUN apt-get install -y libginac-dev libcln-dev

RUN git clone https://github.com/osqp/osqp
RUN cd osqp && mkdir build
RUN cd osqp/build && cmake -G "Unix Makefiles" .. -DOSQP_ALGEBRA_BACKEND=cuda
RUN cd osqp/build && cmake --build . --target install

RUN git clone https://github.com/robotology/osqp-eigen
RUN cd osqp-eigen && mkdir build
RUN cd osqp-eigen/build && cmake ..
RUN cd osqp-eigen/build && make && make install