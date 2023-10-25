cmake_minimum_required(VERSION 3.5)
project(orlqp_di2d)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(orlqp REQUIRED)

list(APPEND CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

include_directories(include)

add_executable(orlqp_di2d_node src/orlqp_di2d_node.cpp)
target_link_libraries(orlqp_di2d_node
  orlqp::orlqp
)
ament_target_dependencies(orlqp_di2d_node
  rclcpp
  sensor_msgs
)

install(TARGETS
  orlqp_di2d_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
