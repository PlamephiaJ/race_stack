#!/bin/bash

source /opt/ros/humble/setup.bash

source ~/ws/install/setup.bash

ros2 launch stack_master base_system_launch.xml \
  sim:=false \
  racecar_version:=BLUE \
  map_name:=track_wmn \
  use_vio:=false "$@"
