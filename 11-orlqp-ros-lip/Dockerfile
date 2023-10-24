FROM ros:humble

RUN apt-get update -y && \
    apt-get install -y g++ cmake git gdb && \
    apt-get install -y libeigen3-dev

RUN git clone https://github.com/osqp/osqp
RUN cd osqp && mkdir build
RUN cd osqp/build && cmake -G "Unix Makefiles" ..
RUN cd osqp/build && cmake --build . --target install

RUN git clone https://github.com/JTylerBoylan/orlqp
RUN cd orlqp && mkdir build
RUN cd orlqp/build && cmake ..
RUN cd orlqp/build && cmake --build . --target install