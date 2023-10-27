cmake_minimum_required(VERSION 3.5)
project(orlqp_wang_lip)

find_package(orlqp REQUIRED)

list(APPEND CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

include_directories(include)

add_executable(main src/main.cpp)
target_link_libraries(main orlqp::orlqp)
