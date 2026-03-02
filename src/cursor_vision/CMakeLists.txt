cmake_minimum_required(VERSION 3.8)
project(cursor_vision)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)

# ---------------------------------------------------------------------------
# Message generation
# ---------------------------------------------------------------------------
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Item.msg"
  "msg/HandlingUnit.msg"
  "msg/PickerEvent.msg"
  "msg/PickerPlan.msg"
  "msg/Detection.msg"
  "msg/DetectionArray.msg"
  "msg/CountResult.msg"
  DEPENDENCIES builtin_interfaces
)

# ---------------------------------------------------------------------------
# Python packages (lib + core)
# ---------------------------------------------------------------------------
ament_python_install_package(lib)
ament_python_install_package(core)

# ---------------------------------------------------------------------------
# Install nodes as executables (strip .py for ros2 run / launch)
# ---------------------------------------------------------------------------
install(PROGRAMS nodes/inventory_node.py
  RENAME inventory_node
  DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS nodes/sap_event_node.py
  RENAME sap_event_node
  DESTINATION lib/${PROJECT_NAME}
)

# ---------------------------------------------------------------------------
# Install config, launch, models
# ---------------------------------------------------------------------------
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models
  OPTIONAL
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
