cmake_minimum_required(VERSION 3.8)
project(xor_package)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Core build system
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Dependencies
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# ---- Interface generation ----
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/XorRequest.srv"
  "msg/XorResult.msg"
  DEPENDENCIES std_msgs
)

install(PROGRAMS
  xor_package/ml_node.py
  xor_package/client_node.py
  xor_package/sub_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# ---- Export runtime dependencies ----
ament_export_dependencies(rosidl_default_runtime)

ament_package()
