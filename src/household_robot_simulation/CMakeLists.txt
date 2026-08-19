cmake_minimum_required(VERSION 3.8)
project(household_robot_simulation)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)

# Install launch files
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# Install world files
install(
  DIRECTORY worlds
  DESTINATION share/${PROJECT_NAME}
)

# Install URDF model files
install(
  DIRECTORY urdf
  DESTINATION share/${PROJECT_NAME}
)

# Install RViz configuration files
install(
  DIRECTORY rviz
  DESTINATION share/${PROJECT_NAME}
)

# Install bridge configuration files
install(
  DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# Install map files
install(
  DIRECTORY maps
  DESTINATION share/${PROJECT_NAME}
)

# Install materials files
install(
  DIRECTORY materials
  DESTINATION share/${PROJECT_NAME}
)

# Install models files
install(
  DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)

# Install python scripts
install(
  PROGRAMS
    scripts/teleop_keyboard.py
    scripts/cmd_vel_safety_filter.py
    scripts/vision_processor.py
    scripts/yolo_detector.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
