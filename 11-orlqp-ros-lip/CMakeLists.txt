cmake_minimum_required(VERSION 3.5)
project(orlqp_lip)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(orlqp REQUIRED)

include_directories(include)

add_executable(orlqp_lip_node src/orlqp_lip_node.cpp)
target_link_libraries(orlqp_lip_node
  orlqp::orlqp
)
ament_target_dependencies(orlqp_lip_node
  rclcpp
  sensor_msgs
)

install(TARGETS
  orlqp_lip_node
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
