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
    src/qp_array_util.cpp
    )
target_include_directories(orlqp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add the executable
add_executable(double_integrator_mpc src/examples/double_integrator_mpc.cpp)
target_include_directories(double_integrator_mpc PUBLIC include)
target_link_libraries(double_integrator_mpc PUBLIC stdc++ stdc++fs m orlqp osqp::osqp)

add_executable(double_integrator_mpc_array src/examples/double_integrator_mpc_array.cpp)
target_include_directories(double_integrator_mpc_array PUBLIC include)
target_link_libraries(double_integrator_mpc_array PUBLIC stdc++ stdc++fs m orlqp osqp::osqp)