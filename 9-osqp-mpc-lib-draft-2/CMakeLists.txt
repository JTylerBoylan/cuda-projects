cmake_minimum_required(VERSION 3.5)
project(orlqp)

find_package(osqp REQUIRED)
find_package(Eigen3)

include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
include_directories(include)

add_library(orlqp SHARED)
target_sources(orlqp PRIVATE
    src/mpc_util.cpp
    src/osqp_util.cpp
    )
target_include_directories(orlqp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add the executable
add_executable(main src/examples/main.cpp)
target_include_directories(main PUBLIC include)
target_link_libraries(main PUBLIC stdc++ stdc++fs m orlqp osqp::osqp)