cmake_minimum_required(VERSION 3.5)
project(orlqp)

find_package(osqp REQUIRED)
find_package(Eigen3)

include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
include_directories(include)

add_library(orlqp SHARED)
target_sources(orlqp PRIVATE
    src/mpc.cpp
    src/osqp.cpp
    src/double_integrator_mpc.cpp
)
target_include_directories(orlqp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add the executable
add_executable(double_integrator_mpc_example src/examples/double_integrator_mpc_example.cpp)
target_include_directories(double_integrator_mpc_example PUBLIC include)
target_link_libraries(double_integrator_mpc_example PUBLIC stdc++ stdc++fs m orlqp osqp::osqp)

# Add the executable
add_executable(multi_double_integrator_mpc_example src/examples/multi_double_integrator_mpc_example.cpp)
target_include_directories(multi_double_integrator_mpc_example PUBLIC include)
target_link_libraries(multi_double_integrator_mpc_example PUBLIC stdc++ stdc++fs m orlqp osqp::osqp)