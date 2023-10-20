cmake_minimum_required(VERSION 3.5)
project(orlqp)

find_package(osqp REQUIRED)
find_package(Eigen3)

include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
include_directories(include)

add_library(orlqp SHARED)
target_sources(orlqp PRIVATE
    src/osqp.cpp
    src/mpc.cpp
    src/qp_array.cpp
    )
target_include_directories(orlqp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add the executable
add_executable(simple_mpc_test tests/simple_mpc.cpp)
target_include_directories(simple_mpc_test PUBLIC include)
target_link_libraries(simple_mpc_test PUBLIC stdc++ stdc++fs m orlqp osqp::osqp)

# Add the executable
add_executable(simple_mpc_array_test tests/simple_mpc_array.cpp)
target_include_directories(simple_mpc_array_test PUBLIC include)
target_link_libraries(simple_mpc_array_test PUBLIC stdc++ stdc++fs m orlqp osqp::osqp)