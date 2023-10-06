cmake_minimum_required(VERSION 3.5)
project(onr_osqp)

set(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(OsqpEigen REQUIRED)

include_directories(include)

# Add the executable
add_executable(main src/main.cpp)
target_include_directories(main PUBLIC include)
target_link_libraries(main PUBLIC stdc++ stdc++fs m OsqpEigen::OsqpEigen)